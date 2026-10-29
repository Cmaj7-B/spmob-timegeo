// TimeGeo parameter estimation — ported per asset/timegeo/README.md §8.
//
// Shell (parquet reading, output layout, CLI) kept from sparkmobility's
// module_2_3_1.cpp. Per-user calibration logic taken from the lab's
// 2-DT_Comm.cpp (timegeo_og) through DT_user_param.cpp's interface.
// Differences from 2-DT_Comm.cpp, each one an existing version's choice:
//   - no forced leave during the grid search (DT_user_param.cpp, Step9CalculateDT.cpp)
//   - AreaTestStat day test uses absolute slot / daily_slot_num, as RescaleTime does
//     in every version (DT_user_param.cpp's correction, completed with the absolute
//     day index the same file uses for day counting)
//   - no user caps (debug limits)
//   - time offset is a parameter, default 0: feed local wall-clock timestamps
//   - grid is a parameter: 0 = paper {1..20} x {1,6,...,101}, 1 = lab {16..20} x {1..5}
//   - RNG seeded from a parameter
//   - P(t) side files written into output_dir
//   - all users are held in memory and processed in two passes (pooled P(t) first,
//     fits second); the previous per-row-group processing split users whose rows
//     crossed a row-group boundary.

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#include <map>
#include <memory>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/resource.h>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;

#define NBINS 10
#define secOneDay 86400

static bool g_quiet = false;
// validation-only switches: reproduce 2-DT_Comm.cpp exactly (forced leave during the search,
// day test that never fires in AreaTestStat) to check the port against the lab binary
static bool g_og_forced_leave = false;
static bool g_og_areastat = false;

struct StayRegion {
    std::string user_id;
    int64_t timestamp = 0;      // seconds
    int location_type = 0;      // 1 = home, 2 = work, 0 = other
    int day_of_week = 0;
    double lat = 0.0;
    double lon = 0.0;
    long long location_id = 0;
};

struct Params {
    bool commuter_mode = true;
    int min_num_stay = 50;
    int max_num_stay = 100000;
    double nw_thres = 1.0;
    int slot_interval = 600;
    double rho = 0.6;
    double gamma = -0.21;
    double week_num_thres = 2.0;
    long tz_offset = 0;         // seconds subtracted from timestamps before slotting
    int grid = 0;               // 0 paper, 1 lab
    long seed = -1;             // <0: time(NULL)
};

// ---------------------------------------------------------------- globals (as in 2-DT_Comm.cpp)
static Params P;
static int period = secOneDay * 7;
static int daily_slot_num = 144;
static int slot_num = 1008;
static int rand_range = 550;
static vector<double> n1_arr, n2_arr;
static vector<double> daily_activeness, weekly_activeness, daily_weekly_activeness;
static vector<double> original_real_dt, rescaled_real_dt, original_simu_dt, rescaled_simu_dt;

static int64_t timeCorrect(int64_t t) { return t - P.tz_offset; }

static double getOtherMovePr(int daily_slot, int weekly_slot, int index, double nw) {
    return n1_arr[index] * nw * daily_activeness[daily_slot] * weekly_activeness[weekly_slot];
}
static double getOtherMoveToHomePr(int daily_slot, int weekly_slot, int index, double nw) {
    return 1 - n2_arr[index] * nw * daily_activeness[daily_slot] * weekly_activeness[weekly_slot];
}

// RescaleTime, as in 2-DT_Comm.cpp (burstiness sums dropped: never written by any version)
static void RescaleTime(const vector<int64_t>& times, bool real_time) {
    vector<double> slot_activeness(slot_num, 0.0);
    int stay_num = times.size();
    for (int i = 0; i < stay_num; i++) {
        int record_slot = (int)((timeCorrect(times[i])) % period / P.slot_interval);
        slot_activeness[record_slot] += 1;
    }
    for (int i = 1; i < stay_num; i++) {
        int64_t slot1 = timeCorrect(times[i - 1]) / P.slot_interval;
        int64_t slot2 = timeCorrect(times[i]) / P.slot_interval;
        int64_t interval_num = (times[i] - times[i - 1]) / P.slot_interval;
        double dt = times[i] - times[i - 1];
        int64_t day1 = slot1 / daily_slot_num;
        int64_t day2 = slot2 / daily_slot_num;
        if (day2 - day1 > 1) {
            continue;  // don't consider intervals that are too long
        }
        double rescaled_time = 0;
        for (int64_t j = slot1; j < slot1 + interval_num; j++) {
            rescaled_time += slot_activeness[j % slot_num];
        }
        if (real_time) {
            original_real_dt.push_back(dt);
            rescaled_real_dt.push_back(rescaled_time);
        } else {
            original_simu_dt.push_back(dt);
            rescaled_simu_dt.push_back(rescaled_time);
        }
    }
}

// AreaTestStat, as in 2-DT_Comm.cpp, with the day test computed as RescaleTime computes it
static double AreaTestStat(const vector<int64_t>& real_time, const vector<int64_t>& simu_time, int n_bins) {
    int m1 = real_time.size();
    int m2 = simu_time.size();
    int real_stay_num = m1;
    int simu_stay_num = m2;
    double days_real = (real_time[m1 - 1] - real_time[0]) / double(secOneDay);
    double days_simu = (simu_time[m2 - 1] - simu_time[0]) / double(secOneDay);
    vector<double> dt_real, dt_simu;
    for (int i = 0; i < m1 - 1; i++) {
        double dt = real_time[i + 1] - real_time[i];
        int64_t day1 = (timeCorrect(real_time[i]) / P.slot_interval) / daily_slot_num;
        int64_t day2 = (timeCorrect(real_time[i + 1]) / P.slot_interval) / daily_slot_num;
        if (g_og_areastat) { day1 = ((timeCorrect(real_time[i]) % period) / P.slot_interval) / slot_num; day2 = ((timeCorrect(real_time[i + 1]) % period) / P.slot_interval) / slot_num; }
        if (day2 - day1 > 1) {
            days_real = days_real - (day2 - day1 - 1);
            continue;
        }
        dt_real.push_back(dt);
    }
    for (int i = 0; i < m2 - 1; i++) {
        double dt = simu_time[i + 1] - simu_time[i];
        int64_t day1 = (timeCorrect(simu_time[i]) / P.slot_interval) / daily_slot_num;
        int64_t day2 = (timeCorrect(simu_time[i + 1]) / P.slot_interval) / daily_slot_num;
        if (g_og_areastat) { day1 = ((timeCorrect(simu_time[i]) % period) / P.slot_interval) / slot_num; day2 = ((timeCorrect(simu_time[i + 1]) % period) / P.slot_interval) / slot_num; }
        if (day2 - day1 > 1) {
            days_simu = days_simu - (day2 - day1 - 1);
            continue;
        }
        dt_simu.push_back(dt);
    }
    m1 = dt_real.size();
    m2 = dt_simu.size();
    if (m1 == 0 || m2 == 0) return 999;
    double min_lnt = 999, max_lnt = -999;
    for (int i = 0; i < m1; i++) {
        max_lnt = max(max_lnt, log(dt_real[i]));
        min_lnt = min(min_lnt, log(dt_real[i]));
    }
    for (int i = 0; i < m2; i++) {
        max_lnt = max(max_lnt, log(dt_simu[i]));
        min_lnt = min(min_lnt, log(dt_simu[i]));
    }
    max_lnt += 0.0001;
    min_lnt -= 0.0001;
    double lnt_range = max_lnt - min_lnt;
    vector<double> real_freq(n_bins, 0.0), simu_freq(n_bins, 0.0);
    for (int i = 0; i < m1; i++) {
        int index = (int)((log(dt_real[i]) - min_lnt) / lnt_range * n_bins);
        if (index >= n_bins) index = n_bins - 1;
        real_freq[index] += 1.00 / m1;
    }
    for (int i = 0; i < m2; i++) {
        int index = (int)((log(dt_simu[i]) - min_lnt) / lnt_range * n_bins);
        if (index >= n_bins) index = n_bins - 1;
        simu_freq[index] += 1.00 / m2;
    }
    double ats = 0;
    for (int i = 0; i < n_bins; i++) ats += fabs(simu_freq[i] - real_freq[i]);
    // N term over days with at least one stay (day_num1/day_num2 in 2-DT_Comm.cpp:346); days_real/days_simu
    // (span minus empty days, Step9CalculateDT.cpp:312) are kept only for the gap bookkeeping above
    int day_num1 = 1, day_num2 = 1;
    for (int dd = 1; dd < real_stay_num; dd++) if (timeCorrect(real_time[dd - 1]) / secOneDay != timeCorrect(real_time[dd]) / secOneDay) day_num1++;
    for (int dd = 1; dd < simu_stay_num; dd++) if (timeCorrect(simu_time[dd - 1]) / secOneDay != timeCorrect(simu_time[dd]) / secOneDay) day_num2++;
    ats += 0.035 * fabs((real_stay_num + 0.0) / day_num1 - (simu_stay_num + 0.0) / day_num2);
    return ats;
}

// ---------------------------------------------------------------- per-user structures
struct UserSeq {
    vector<int64_t> times;
    vector<int> locs;
    vector<long long> locids;
    vector<double> lons, lats;
    bool has_work = false;
};

static bool passes_filters(const UserSeq& u) {
    int stay_num = u.times.size();
    if (!(stay_num > P.min_num_stay && stay_num < P.max_num_stay && u.has_work == P.commuter_mode)) return false;
    double week_num = (u.times[stay_num - 1] - u.times[0]) / 604800;   // integer division, as in 2-DT_Comm.cpp
    if (week_num < P.week_num_thres) return false;
    for (int i = 0; i < stay_num - 1; i++) {
        if (u.times[i + 1] <= u.times[i]) return false;
    }
    return true;
}

// Pass 1: pooled circadian profile from non-work stays of all valid users (2-DT_Comm.cpp 470-515)
static void accumulate_activeness(const UserSeq& u, long& activity_count) {
    for (size_t i = 0; i < u.times.size(); i++) {
        if (u.locs[i] != 2) {
            int daily_slot = (int)((timeCorrect(u.times[i])) % secOneDay / P.slot_interval);
            int weekly_slot = ((int)((timeCorrect(u.times[i])) / secOneDay)) % 7;
            daily_activeness[daily_slot] += 1;
            weekly_activeness[weekly_slot] += 1;
            daily_weekly_activeness[weekly_slot * daily_slot_num + daily_slot] += 1;
            activity_count++;
        }
    }
}

// One Markov simulation over the user's observation window (2-DT_Comm.cpp 679-783, without pt = 100000)
static void simulate_user(const UserSeq& u, double nw, int index1, int index2,
                          const vector<int64_t>& work_start, const vector<int64_t>& work_end,
                          vector<int64_t>& simulated_time, vector<int>& simulated_at_home, bool forced_leave = false) {
    int stay_num = u.times.size();
    int64_t current_time = u.times[0];
    int64_t end_time = u.times[stay_num - 1];
    simulated_time.clear();
    simulated_at_home.clear();
    bool at_home = true, at_work = false, need_move = false;
    int current_work_index = 0;
    int max_work_index = (int)work_start.size() - 1;
    simulated_time.push_back(current_time);
    simulated_at_home.push_back(1);
    while (current_time < end_time) {
        current_time += P.slot_interval + rand() % rand_range - ((int)(0.5 * rand_range));
        if (P.commuter_mode && current_work_index > max_work_index) {
            need_move = false;
            at_work = false;
        }
        if (P.commuter_mode && current_work_index <= max_work_index) {
            need_move = false;
            if (current_time >= work_end[current_work_index]) {
                current_work_index++;
                if (at_work) need_move = true;
                at_work = false;
            }
            if (current_work_index <= max_work_index && current_time > work_start[current_work_index]) {
                if (!at_work) {
                    simulated_time.push_back(current_time);
                    at_home = false;
                    at_work = true;
                    simulated_at_home.push_back(2);
                    continue;
                }
            }
        }
        int daily_slot = (int)((timeCorrect(current_time)) % secOneDay / P.slot_interval);
        int weekly_slot = ((int)((timeCorrect(current_time)) / secOneDay)) % 7;
        if (at_home) {
            double pt = nw * daily_activeness[daily_slot] * weekly_activeness[weekly_slot];
            if (forced_leave) pt = 100000;
            if ((rand() % 100000) / 100000.0 < pt) {
                simulated_time.push_back(current_time);
                at_home = false;
                simulated_at_home.push_back(0);
            }
        } else {
            double p_other_move = need_move ? 1.0 : getOtherMovePr(daily_slot, weekly_slot, index1, nw);
            if ((rand() % 100000) / 100000.0 < p_other_move) {
                simulated_time.push_back(current_time);
                double p_other_home = getOtherMoveToHomePr(daily_slot, weekly_slot, index2, nw);
                if ((rand() % 100000) / 100000.0 < p_other_home) {
                    at_home = true;
                    simulated_at_home.push_back(1);
                } else {
                    simulated_at_home.push_back(0);
                }
            }
        }
    }
}

// Daily counts: (stays per day, distinct location-type visits per day), with zero lines for gap days (2-DT_Comm.cpp 915-1064)
static void write_daily_counts(FILE* f, const vector<int64_t>& times, const vector<int>& locs,
                               double* total_daily_loc_count, double* total_day) {
    int n = times.size();
    if (n == 0) return;
    int64_t current_day = timeCorrect(times[0]) / secOneDay;
    int daily_count = 1, daily_location_count = 1;
    bool was_home = (locs[0] == 1), was_work = (locs[0] == 2);
    for (int i = 1; i < n; i++) {
        int64_t previous_day = current_day;
        current_day = timeCorrect(times[i]) / secOneDay;
        if (current_day == previous_day) {
            daily_count++;
            if (was_home && locs[i] == 1) {
            } else if (was_work && locs[i] == 2) {
            } else {
                daily_location_count++;
            }
            if (locs[i] == 1) was_home = true;
            if (locs[i] == 2) was_work = true;
        } else {
            int64_t gap_days = current_day - previous_day - 1;
            for (int64_t j = 0; j < gap_days; j++) fprintf(f, "%d %d\n", 0, 1);
            fprintf(f, "%d %d\n", daily_count, daily_location_count);
            if (total_daily_loc_count) { *total_daily_loc_count += daily_location_count; *total_day += 1; }
            daily_count = 1;
            daily_location_count = 1;
            was_home = (locs[i] == 1);
            was_work = (locs[i] == 2);
        }
    }
}

// Pass 2: calibrate one user (2-DT_Comm.cpp 572-1145)
static void fit_user(const string& uid, const UserSeq& u,
                     FILE* fout_dn_real, FILE* fout_dn_simu, FILE* fout_simuloc, FILE* fout_params, FILE* fout_realloc) {
    int stay_num = u.times.size();
    double work_duration = 0;
    double total_duration = u.times[stay_num - 1] - u.times[0];

    double home_lon = 0, home_lat = 0, work_lon = 0, work_lat = 0;
    for (int i = 0; i < stay_num; i++) if (u.locs[i] == 1) { home_lon = u.lons[i]; home_lat = u.lats[i]; break; }
    for (int i = 0; i < stay_num; i++) if (u.locs[i] == 2) { work_lon = u.lons[i]; work_lat = u.lats[i]; break; }

    int day_num = 1;
    int home_based_trip = 1;
    for (int dd = 1; dd < stay_num; dd++) {
        int64_t prev_day = timeCorrect(u.times[dd - 1]) / secOneDay;
        int64_t curr_day = timeCorrect(u.times[dd]) / secOneDay;
        if (prev_day != curr_day) day_num++;
    }
    vector<int64_t> work_start, work_end;
    for (int dd = 0; dd < stay_num - 1; dd++) {
        if (u.locs[dd] == 1 && u.locs[dd + 1] != 2) home_based_trip++;
        if (u.locs[dd] == 2) {
            work_start.push_back(u.times[dd]);
            work_end.push_back(u.times[dd + 1]);
            work_duration += u.times[dd + 1] - u.times[dd];
        }
    }
    double nw = home_based_trip / (day_num / 7.000);
    if (P.commuter_mode && total_duration > work_duration) {
        nw = nw * total_duration / (total_duration - work_duration);
    }
    if (nw < P.nw_thres) return;

    // grid search
    double best_ats = 999;
    int best_index1 = 0, best_index2 = 0;
    vector<int64_t> simulated_time;
    vector<int> simulated_at_home;
    for (size_t index1 = 0; index1 < n1_arr.size(); index1++) {
        for (size_t index2 = 0; index2 < n2_arr.size(); index2++) {
            simulate_user(u, nw, index1, index2, work_start, work_end, simulated_time, simulated_at_home, g_og_forced_leave);
            double ats = AreaTestStat(u.times, simulated_time, NBINS);
            if (ats < best_ats) {
                best_ats = ats;
                best_index1 = index1;
                best_index2 = index2;
            }
        }
    }
    // run again with the best parameters (2-DT_Comm.cpp 803-910)
    vector<int64_t> best_simu_time;
    vector<int> best_simu_at_home;
    simulate_user(u, nw, best_index1, best_index2, work_start, work_end, best_simu_time, best_simu_at_home);

    // daily counts
    write_daily_counts(fout_dn_simu, best_simu_time, best_simu_at_home, nullptr, nullptr);
    double total_daily_loc_count = 0, total_day = 0;
    write_daily_counts(fout_dn_real, u.times, u.locs, &total_daily_loc_count, &total_day);

    // empirical records
    for (int i = 0; i < stay_num; i++) {
        fprintf(fout_realloc, "%s %lld %d %f %lld\n", uid.c_str(), (long long)u.times[i], u.locs[i],
                double(timeCorrect(u.times[i])) / secOneDay, u.locids[i]);
    }

    // simulated records with location ids from preferential return (2-DT_Comm.cpp 1071-1140)
    int visited_location = 0;
    vector<int> visit_count;
    vector<double> visit_cdf;
    int prev_loc = -1;
    int stay_num_simu = best_simu_time.size();
    for (int i = 1; i < stay_num_simu; i++) {
        double day = double(timeCorrect(best_simu_time[i])) / secOneDay;
        if (best_simu_at_home[i] == 1) {
            fprintf(fout_simuloc, "%s %lld %d %f %d\n", uid.c_str(), (long long)best_simu_time[i], 1, day, 0);
            prev_loc = 0;
        } else if (best_simu_at_home[i] == 2) {
            fprintf(fout_simuloc, "%s %lld %d %f %d\n", uid.c_str(), (long long)best_simu_time[i], 2, day, 1);
            prev_loc = 0;
        } else {
            if (visited_location < 2) {
                visited_location++;
                fprintf(fout_simuloc, "%s %lld %d %f %d\n", uid.c_str(), (long long)best_simu_time[i], 0, day, 2 + visited_location);
                visit_count.push_back(1);
                prev_loc = 1;
            } else {
                double r_return_unconditioned = 0.6 * (1 - P.rho * pow(visited_location, P.gamma));
                double p_return = r_return_unconditioned / (r_return_unconditioned + P.rho * pow(visited_location, P.gamma));
                if ((rand() % 9999999) / 9999999.0 < p_return) {
                    double visit_count_sum = 0;
                    for (int j = 0; j < visited_location; j++) visit_count_sum += visit_count[j];
                    visit_cdf.clear();
                    visit_cdf.push_back(visit_count[0] / visit_count_sum);
                    for (int j = 1; j < visited_location; j++) visit_cdf.push_back(visit_cdf[j - 1] + visit_count[j] / visit_count_sum);
                    bool found_sign = false;
                    int guard = 0;
                    while (!found_sign && guard++ < 100000) {
                        double rand_num = (rand() % 9999999) / 9999999.0;
                        for (int j = 0; j < visited_location; j++) {
                            if (rand_num < visit_cdf[j] && j != prev_loc - 1) {
                                ++visit_count[j];
                                fprintf(fout_simuloc, "%s %lld %d %f %d\n", uid.c_str(), (long long)best_simu_time[i], 0, day, j + 1 + 2);
                                prev_loc = j + 1;
                                found_sign = true;
                                break;
                            }
                        }
                    }
                } else {
                    visited_location++;
                    fprintf(fout_simuloc, "%s %lld %d %f %d\n", uid.c_str(), (long long)best_simu_time[i], 0, day, visited_location + 2);
                    visit_count.push_back(1);
                }
            }
        }
    }

    fprintf(fout_params, "%d %d %f %f %f %f %f %f %s\n", best_index1, best_index2, nw,
            total_day > 0 ? total_daily_loc_count / total_day * 1.3 : 0.0,
            home_lon, home_lat, work_lon, work_lat, uid.c_str());

    RescaleTime(u.times, true);
    RescaleTime(best_simu_time, false);
}

// ---------------------------------------------------------------- parquet input (shell kept from module_2_3_1.cpp)
static vector<string> get_parquet_files(const string& path) {
    vector<string> parquet_files;
    struct stat path_stat;
    stat(path.c_str(), &path_stat);
    if (S_ISDIR(path_stat.st_mode)) {
        DIR* dir = opendir(path.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL) {
                string file_name = entry->d_name;
                if (file_name.find(".parquet") != string::npos && file_name.find(".crc") == string::npos)
                    parquet_files.push_back(path + "/" + file_name);
            }
            closedir(dir);
        }
    } else if (path.find(".parquet") != string::npos) {
        parquet_files.push_back(path);
    }
    sort(parquet_files.begin(), parquet_files.end());
    return parquet_files;
}

static void read_parquet_into(const string& input_path, unordered_map<string, vector<StayRegion>>& user_data, long& total_records) {
    auto memory_pool = arrow::default_memory_pool();
    auto maybe_infile = arrow::io::ReadableFile::Open(input_path, memory_pool);
    if (!maybe_infile.ok()) { cerr << "Could not open file: " << input_path << endl; return; }
    shared_ptr<arrow::io::ReadableFile> infile = maybe_infile.ValueOrDie();
    auto maybe_reader = parquet::arrow::OpenFile(infile, memory_pool);
    if (!maybe_reader.ok()) { cerr << "Could not open Parquet file: " << input_path << " - " << maybe_reader.status().ToString() << endl; return; }
    unique_ptr<parquet::arrow::FileReader> reader = move(maybe_reader).ValueOrDie();
    int num_row_groups = reader->parquet_reader()->metadata()->num_row_groups();
    for (int group = 0; group < num_row_groups; group++) {
        shared_ptr<arrow::Table> table;
        auto status = reader->ReadRowGroup(group, &table);
        if (!status.ok()) { cerr << "Error reading row group " << group << ": " << status.ToString() << endl; continue; }
        int user_id_idx = -1, timestamp_idx = -1, type_idx = -1, day_idx = -1, lat_idx = -1, lon_idx = -1, locid_idx = -1;
        for (int i = 0; i < table->schema()->num_fields(); i++) {
            string f = table->schema()->field(i)->name();
            transform(f.begin(), f.end(), f.begin(), ::tolower);
            if (f == "caid" || f == "user_id" || f == "userid") user_id_idx = i;
            else if (f == "stay_start_timestamp" || f == "timestamp") timestamp_idx = i;
            else if (f == "type") type_idx = i;
            else if (f == "day_of_week" || f == "day") day_idx = i;
            else if (f == "latitude" || f == "lat") lat_idx = i;
            else if (f == "longitude" || f == "lon") lon_idx = i;
            else if (f == "h3_region_stay_id" || f == "location_id") locid_idx = i;
        }
        if (user_id_idx == -1 || timestamp_idx == -1) { cerr << "Required columns not found" << endl; continue; }
        int num_chunks = table->column(user_id_idx)->num_chunks();
        for (int c = 0; c < num_chunks; c++) {
            auto user_id_array = table->column(user_id_idx)->chunk(c);
            auto timestamp_array = table->column(timestamp_idx)->chunk(c);
            auto type_array = type_idx >= 0 ? table->column(type_idx)->chunk(c) : nullptr;
            auto day_array = day_idx >= 0 ? table->column(day_idx)->chunk(c) : nullptr;
            auto lat_array = lat_idx >= 0 ? table->column(lat_idx)->chunk(c) : nullptr;
            auto lon_array = lon_idx >= 0 ? table->column(lon_idx)->chunk(c) : nullptr;
            auto locid_array = locid_idx >= 0 ? table->column(locid_idx)->chunk(c) : nullptr;
            auto s_uid = dynamic_pointer_cast<arrow::StringArray>(user_id_array);
            auto i32_uid = dynamic_pointer_cast<arrow::Int32Array>(user_id_array);
            auto i64_uid = dynamic_pointer_cast<arrow::Int64Array>(user_id_array);
            auto ts_ts = dynamic_pointer_cast<arrow::TimestampArray>(timestamp_array);
            auto ts_i64 = dynamic_pointer_cast<arrow::Int64Array>(timestamp_array);
            auto ts_i32 = dynamic_pointer_cast<arrow::Int32Array>(timestamp_array);
            int64_t ts_div = 1;
            if (ts_ts) {
                auto unit = static_pointer_cast<arrow::TimestampType>(ts_ts->type())->unit();
                ts_div = unit == arrow::TimeUnit::NANO ? 1000000000LL : unit == arrow::TimeUnit::MICRO ? 1000000LL : unit == arrow::TimeUnit::MILLI ? 1000LL : 1LL;
            }
            auto type_i32 = type_array ? dynamic_pointer_cast<arrow::Int32Array>(type_array) : nullptr;
            auto type_i64 = type_array ? dynamic_pointer_cast<arrow::Int64Array>(type_array) : nullptr;
            auto day_i32 = day_array ? dynamic_pointer_cast<arrow::Int32Array>(day_array) : nullptr;
            auto lat_d = lat_array ? dynamic_pointer_cast<arrow::DoubleArray>(lat_array) : nullptr;
            auto lon_d = lon_array ? dynamic_pointer_cast<arrow::DoubleArray>(lon_array) : nullptr;
            auto locid_i32 = locid_array ? dynamic_pointer_cast<arrow::Int32Array>(locid_array) : nullptr;
            auto locid_i64 = locid_array ? dynamic_pointer_cast<arrow::Int64Array>(locid_array) : nullptr;
            int64_t n = user_id_array->length();
            for (int64_t i = 0; i < n; i++) {
                if (user_id_array->IsNull(i) || timestamp_array->IsNull(i)) continue;
                StayRegion r;
                if (s_uid) r.user_id = s_uid->GetString(i);
                else if (i32_uid) r.user_id = to_string(i32_uid->Value(i));
                else if (i64_uid) r.user_id = to_string(i64_uid->Value(i));
                else continue;
                if (ts_ts) r.timestamp = ts_ts->Value(i) / ts_div;
                else if (ts_i64) r.timestamp = ts_i64->Value(i);
                else if (ts_i32) r.timestamp = ts_i32->Value(i);
                else continue;
                if (type_array && !type_array->IsNull(i)) {
                    if (type_i32) r.location_type = type_i32->Value(i);
                    else if (type_i64) r.location_type = (int)type_i64->Value(i);
                }
                if (day_i32 && !day_array->IsNull(i)) r.day_of_week = day_i32->Value(i);
                if (lat_d && !lat_array->IsNull(i)) r.lat = lat_d->Value(i);
                if (lon_d && !lon_array->IsNull(i)) r.lon = lon_d->Value(i);
                if (locid_array && !locid_array->IsNull(i)) {
                    if (locid_i32) r.location_id = locid_i32->Value(i);
                    else if (locid_i64) r.location_id = locid_i64->Value(i);
                }
                user_data[r.user_id].push_back(r);
                total_records++;
            }
        }
    }
}

// ---------------------------------------------------------------- driver
static void set_grid(int grid) {
    n1_arr.clear(); n2_arr.clear();
    if (grid == 1) {
        n1_arr = {16, 17, 18, 19, 20};
        n2_arr = {1, 2, 3, 4, 5};
    } else {
        for (int v = 1; v <= 20; v++) n1_arr.push_back(v);
        for (int v = 1; v <= 101; v += 5) n2_arr.push_back(v);
    }
}

static void run(const string& input_path, const string& output_dir) {
    daily_slot_num = secOneDay / P.slot_interval;
    slot_num = period / P.slot_interval;
    rand_range = P.slot_interval - 50;
    set_grid(P.grid);
    srand(P.seed < 0 ? (unsigned)time(NULL) : (unsigned)P.seed);

    string sub = P.commuter_mode ? "/Commuters/" : "/NonCommuters/";
    string tag = P.commuter_mode ? "Commuters" : "NonCommuters";
    string pt_tag = P.commuter_mode ? "Comm" : "NonComm";
    fs::create_directories(output_dir + sub);
    FILE* f_dtreal = fopen((output_dir + sub + "DTReal" + tag + ".txt").c_str(), "w");
    FILE* f_dtsimu = fopen((output_dir + sub + "DTSimu" + tag + ".txt").c_str(), "w");
    FILE* f_dnreal = fopen((output_dir + sub + "DNReal" + tag + ".txt").c_str(), "w");
    FILE* f_dnsimu = fopen((output_dir + sub + "DNSimu" + tag + ".txt").c_str(), "w");
    FILE* f_simuloc = fopen((output_dir + sub + "SimuLoc" + tag + ".txt").c_str(), "w");
    FILE* f_params = fopen((output_dir + sub + "Parameters" + tag + ".txt").c_str(), "w");
    FILE* f_realloc = fopen((output_dir + sub + "RealLoc" + tag + ".txt").c_str(), "w");
    FILE* f_grid = fopen((output_dir + sub + "Grid" + tag + ".txt").c_str(), "w");
    for (double v : n1_arr) fprintf(f_grid, "%g ", v); fprintf(f_grid, "\n");
    for (double v : n2_arr) fprintf(f_grid, "%g ", v); fprintf(f_grid, "\n");
    fclose(f_grid);

    // read everything (all users complete), then two passes
    unordered_map<string, vector<StayRegion>> user_data;
    long total_records = 0;
    for (const auto& file : get_parquet_files(input_path)) read_parquet_into(file, user_data, total_records);
    if (!g_quiet) cout << "read " << total_records << " records for " << user_data.size() << " users" << endl;

    map<string, UserSeq> users;   // ordered: deterministic output order
    for (auto& [uid, regions] : user_data) {
        sort(regions.begin(), regions.end(), [](const StayRegion& a, const StayRegion& b) { return a.timestamp < b.timestamp; });
        UserSeq u;
        for (const auto& r : regions) {
            u.times.push_back(r.timestamp);
            u.locs.push_back(r.location_type);
            u.locids.push_back(r.location_id);
            u.lons.push_back(r.lon);
            u.lats.push_back(r.lat);
            if (r.location_type == 2) u.has_work = true;
        }
        users.emplace(uid, move(u));
    }
    user_data.clear();

    // pass 1
    daily_activeness.assign(daily_slot_num, 0.0);
    weekly_activeness.assign(7, 0.0);
    daily_weekly_activeness.assign(7 * daily_slot_num, 0.0);
    long activity_count = 0;
    long n_valid = 0;
    for (const auto& [uid, u] : users) {
        if (!passes_filters(u)) continue;
        n_valid++;
        accumulate_activeness(u, activity_count);
    }
    if (activity_count > 0) {
        for (auto& v : daily_activeness) v /= activity_count;
        for (auto& v : weekly_activeness) v /= activity_count;
        for (auto& v : daily_weekly_activeness) v /= activity_count;
    }
    {
        FILE* fd = fopen((output_dir + "/" + pt_tag + "_pt_daily.txt").c_str(), "w");
        FILE* fw = fopen((output_dir + "/" + pt_tag + "_pt_weekly.txt").c_str(), "w");
        FILE* fdw = fopen((output_dir + "/" + pt_tag + "_pt_daily_weekly.txt").c_str(), "w");
        for (double v : daily_activeness) fprintf(fd, "%f\n", v);
        for (double v : weekly_activeness) fprintf(fw, "%f\n", v);
        for (double v : daily_weekly_activeness) fprintf(fdw, "%f\n", v);
        fclose(fd); fclose(fw); fclose(fdw);
    }
    cout << "pass 1: " << n_valid << " of " << users.size() << " users pass the filters; " << activity_count << " non-work stays in P(t)" << endl;

    // pass 2
    long n_fit = 0;
    for (const auto& [uid, u] : users) {
        if (!passes_filters(u)) continue;
        long before = ftell(f_params);
        fit_user(uid, u, f_dnreal, f_dnsimu, f_simuloc, f_params, f_realloc);
        if (ftell(f_params) != before) n_fit++;
    }
    for (size_t i = 0; i < original_real_dt.size(); i++) fprintf(f_dtreal, "%f %f\n", original_real_dt[i], rescaled_real_dt[i]);
    for (size_t i = 0; i < original_simu_dt.size(); i++) fprintf(f_dtsimu, "%f %f\n", original_simu_dt[i], rescaled_simu_dt[i]);
    fclose(f_dtreal); fclose(f_dtsimu); fclose(f_dnreal); fclose(f_dnsimu); fclose(f_simuloc); fclose(f_params); fclose(f_realloc);
    cout << "Total users processed: " << n_fit << endl;
}

int main(int argc, char* argv[]) {
    vector<string> pos;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--quiet") { g_quiet = true; continue; }
        if (a == "--verbose") { g_quiet = false; continue; }
        if (a == "--og-forced-leave") { g_og_forced_leave = true; continue; }
        if (a == "--og-areastat") { g_og_areastat = true; continue; }
        pos.push_back(a);
    }
    if (const char* env = getenv("TIMEGEO_QUIET")) if (string(env) == "1") g_quiet = true;
    if (pos.size() < 2) {
        cerr << "usage: module_2_3_1 <input.parquet|dir> <output_dir> [commuter 0|1] [min_num_stay] [max_num_stay] [nw_thres] [slot_interval] [rho] [gamma] [week_num_thres] [tz_offset_s] [grid 0=paper|1=lab] [seed]" << endl;
        return 1;
    }
    long mem_limit_mb = 16384L;
    if (const char* env = getenv("SPARKMOBILITY_MEM_LIMIT_MB")) { long v = atol(env); if (v >= 512L) mem_limit_mb = v; }
    struct rlimit rl;
    rl.rlim_cur = rl.rlim_max = (rlim_t)mem_limit_mb * 1024L * 1024L;
    setrlimit(RLIMIT_AS, &rl);

    string input_path = pos[0];
    string output_dir = pos[1];
    P.commuter_mode = pos.size() > 2 ? (stoi(pos[2]) != 0) : true;
    P.min_num_stay = pos.size() > 3 ? stoi(pos[3]) : 50;
    P.max_num_stay = pos.size() > 4 ? stoi(pos[4]) : 100000;
    P.nw_thres = pos.size() > 5 ? stod(pos[5]) : 1.0;
    P.slot_interval = pos.size() > 6 ? stoi(pos[6]) : 600;
    P.rho = pos.size() > 7 ? stod(pos[7]) : 0.6;
    P.gamma = pos.size() > 8 ? stod(pos[8]) : -0.21;
    P.week_num_thres = pos.size() > 9 ? stod(pos[9]) : 2.0;
    P.tz_offset = pos.size() > 10 ? stol(pos[10]) : 0;
    P.grid = pos.size() > 11 ? stoi(pos[11]) : 0;
    P.seed = pos.size() > 12 ? stol(pos[12]) : -1;
    run(input_path, output_dir);
    cout << "Processing completed successfully." << endl;
    return 0;
}
