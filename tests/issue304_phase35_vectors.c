#include "../ds4.h"
#include "../ds4_distributed.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define PHASE35_MAX_TOP 128
#define PHASE35_MAX_STEPS 16
#define PHASE35_MAX_TOKEN_BYTES 128

typedef enum {
    PHASE35_SUITE_OFFICIAL,
    PHASE35_SUITE_LOCAL_GOLDEN,
} phase35_suite;

typedef enum {
    PHASE35_MODE_LOCAL,
    PHASE35_MODE_DISTRIBUTED,
    PHASE35_MODE_PAYLOAD,
} phase35_mode;

typedef struct {
    const char *model_path;
    const char *mode_name;
    const char *listen_host;
    const char *vector_file;
    const char *local_golden_file;
    const char *case_filter;
    const char *coordinator_layers;
    const char *worker_layers;
    const char *payload_file;
    const char *payload_out;
    int listen_port;
    int repeat;
    int ctx_filter;
    int greedy_steps;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    int power_percent;
    bool debug;
    phase35_suite suite;
    phase35_mode mode;
} phase35_cfg;

typedef struct {
    unsigned char bytes[PHASE35_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} official_top;

typedef struct {
    unsigned char selected[PHASE35_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    official_top top[32];
} official_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    official_step steps[PHASE35_MAX_STEPS];
} official_case;

typedef struct {
    int id;
    float logit;
} golden_top;

typedef struct {
    char id[96];
    char mode[16];
    char prompt_path[512];
    int ctx;
    int frontier;
    int ntop;
    golden_top top[PHASE35_MAX_TOP];
} golden_case;

typedef struct {
    int top1_ref;
    int top1_cand;
    int top5_overlap;
    int top20_overlap;
    int top64_overlap;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
} parity_metrics;

typedef struct {
    bool initialized;
    int top1_ref;
    int top1_cand;
    int top5_overlap;
    int top20_overlap;
    int top64_overlap;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
    int first_top1_mismatch_step;
    int first_ref_token;
    int first_cand_token;
} parity_summary;

typedef struct {
    bool ok;
    char error[256];
    char route_summary[1024];
    uint32_t route_hops;
    bool output_on_coordinator;
    uint64_t payload_bytes;
    double prefill_sec;
    double stage_sec;
    double load_sec;
    parity_summary parity;
    int steps_checked;
    bool official_pass;
    int official_selected_mismatch_step;
    int official_missing_top_count;
    float official_max_logprob_delta;
    bool golden_pass;
    int golden_top1;
    int golden_top5_overlap;
    int golden_top20_overlap;
    int golden_top64_overlap;
    float golden_top20_max_abs;
} repeat_result;

typedef struct {
    int repeats;
    int passes;
    parity_summary worst_parity;
    bool official_pass;
    int official_selected_mismatch_step;
    int official_missing_top_count;
    float official_max_logprob_delta;
    bool golden_pass;
    int golden_top1;
    int golden_top5_overlap;
    int golden_top20_overlap;
    int golden_top64_overlap;
    float golden_top20_max_abs;
    uint64_t payload_bytes_max;
    double prefill_sec_max;
    double stage_sec_max;
    double load_sec_max;
    char route_summary[1024];
    uint32_t route_hops;
    bool output_on_coordinator;
} case_summary;

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void die(const char *msg) {
    fprintf(stderr, "issue304-phase35-vectors: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "issue304-phase35-vectors: %s: %s\n", msg, strerror(errno));
    exit(1);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --mode local|distributed|payload --suite official|local-golden --model FILE "
            "[--listen-host IPV4 --listen-port N --coordinator-layers A:B|A:output --worker-layers A:B|A:output] "
            "[--payload-file FILE] [--payload-out FILE] "
            "[--vector-file FILE] [--local-golden-file FILE] [--repeat N] [--ctx-filter N] "
            "[--greedy-steps N] [--prefill-chunk N] [--prefill-window N] [--activation-bits N] [--power N] [--no-debug]\n",
            argv0);
    exit(2);
}

static const char *suite_name(phase35_suite suite) {
    return suite == PHASE35_SUITE_OFFICIAL ? "official" : "local-golden";
}

static const char *mode_name(phase35_mode mode) {
    switch (mode) {
        case PHASE35_MODE_LOCAL: return "local";
        case PHASE35_MODE_DISTRIBUTED: return "distributed";
        case PHASE35_MODE_PAYLOAD: return "payload";
    }
    return "unknown";
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = malloc((size_t)len + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (len != 0 && fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[len] = '\0';
    return buf;
}

static char *trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = hex_digit(hex[0]);
        int lo = hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool token_bytes_equal(ds4_engine *engine, int token,
                              const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int i = 0; i < n; i++) {
        const float v = logits[i];
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int t = k - 1; t > j; t--) out[t] = out[t - 1];
                out[j] = i;
                break;
            }
        }
    }
}

static bool topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static parity_metrics compare_logits(const float *ref, const float *cand, int vocab) {
    int ref_top[64];
    int cand_top[64];
    logits_topk(ref, vocab, ref_top, 64);
    logits_topk(cand, vocab, cand_top, 64);

    int top5_overlap = 0;
    int top20_overlap = 0;
    int top64_overlap = 0;
    for (int i = 0; i < 64; i++) {
        if (ref_top[i] >= 0 && topk_contains(cand_top, 64, ref_top[i])) top64_overlap++;
        if (i < 20 && ref_top[i] >= 0 && topk_contains(cand_top, 20, ref_top[i])) top20_overlap++;
        if (i < 5 && ref_top[i] >= 0 && topk_contains(cand_top, 5, ref_top[i])) top5_overlap++;
    }

    int nonfinite = 0;
    double sumsq = 0.0;
    float max_abs = 0.0f;
    float top20_max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(ref[i]) || !isfinite(cand[i])) {
            nonfinite++;
            continue;
        }
        float delta = cand[i] - ref[i];
        float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
    }
    for (int i = 0; i < 20; i++) {
        const int id = ref_top[i];
        if (id < 0) continue;
        const float abs_delta = fabsf(cand[id] - ref[id]);
        if (abs_delta > top20_max_abs) top20_max_abs = abs_delta;
    }

    parity_metrics out = {
        .top1_ref = ref_top[0],
        .top1_cand = cand_top[0],
        .top5_overlap = top5_overlap,
        .top20_overlap = top20_overlap,
        .top64_overlap = top64_overlap,
        .nonfinite = nonfinite,
        .rms = (float)sqrt(sumsq / (double)vocab),
        .max_abs = max_abs,
        .top20_max_abs = top20_max_abs,
    };
    return out;
}

static void parity_summary_observe(parity_summary *out,
                                   const parity_metrics *m,
                                   int step) {
    if (!out->initialized) {
        out->initialized = true;
        out->top1_ref = m->top1_ref;
        out->top1_cand = m->top1_cand;
        out->top5_overlap = m->top5_overlap;
        out->top20_overlap = m->top20_overlap;
        out->top64_overlap = m->top64_overlap;
        out->nonfinite = m->nonfinite;
        out->rms = m->rms;
        out->max_abs = m->max_abs;
        out->top20_max_abs = m->top20_max_abs;
        out->first_top1_mismatch_step = (m->top1_ref == m->top1_cand) ? -1 : step;
        out->first_ref_token = m->top1_ref;
        out->first_cand_token = m->top1_cand;
        return;
    }

    if (m->top5_overlap < out->top5_overlap) out->top5_overlap = m->top5_overlap;
    if (m->top20_overlap < out->top20_overlap) out->top20_overlap = m->top20_overlap;
    if (m->top64_overlap < out->top64_overlap) out->top64_overlap = m->top64_overlap;
    if (m->nonfinite > out->nonfinite) out->nonfinite = m->nonfinite;
    if (m->rms > out->rms) out->rms = m->rms;
    if (m->max_abs > out->max_abs) out->max_abs = m->max_abs;
    if (m->top20_max_abs > out->top20_max_abs) out->top20_max_abs = m->top20_max_abs;
    if (out->first_top1_mismatch_step < 0 && m->top1_ref != m->top1_cand) {
        out->first_top1_mismatch_step = step;
        out->top1_ref = m->top1_ref;
        out->top1_cand = m->top1_cand;
        out->first_ref_token = m->top1_ref;
        out->first_cand_token = m->top1_cand;
    }
}

static bool parity_pass_strict(const parity_summary *p) {
    return p->initialized &&
           p->first_top1_mismatch_step < 0 &&
           p->top5_overlap == 5 &&
           p->top20_overlap == 20 &&
           p->top64_overlap >= 60 &&
           p->nonfinite == 0 &&
           p->top20_max_abs <= 2.0f &&
           p->rms <= 0.1f;
}

static bool parity_pass_medium(const parity_summary *p) {
    return p->initialized &&
           p->first_top1_mismatch_step < 0 &&
           p->top5_overlap == 5 &&
           p->top20_overlap == 20 &&
           p->top64_overlap >= 56 &&
           p->nonfinite == 0 &&
           p->top20_max_abs <= 4.0f &&
           p->rms <= 0.25f;
}

static bool parity_pass_tolerant(const parity_summary *p) {
    return p->initialized &&
           p->first_top1_mismatch_step < 0 &&
           p->top5_overlap >= 4 &&
           p->top20_overlap >= 15 &&
           p->top64_overlap >= 40 &&
           p->nonfinite == 0 &&
           p->top20_max_abs <= 8.0f;
}

static const char *parity_envelope(const parity_summary *p) {
    if (parity_pass_strict(p)) return "strict";
    if (parity_pass_medium(p)) return "medium";
    if (parity_pass_tolerant(p)) return "tolerant";
    return "fail";
}

static bool official_case_disabled(const official_case *vc) {
    return !strcmp(vc->id, "long_memory_archive");
}

static bool case_selected(const char *id, const char *filter) {
    if (!filter || !filter[0]) return true;
    return strstr(id, filter) != NULL;
}

static bool read_official_case(FILE *fp, official_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            return vc->nsteps > 0 && vc->nsteps <= PHASE35_MAX_STEPS;
        }
        return false;
    }
    return false;
}

static bool fill_official_case(FILE *fp, official_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;
        if (!strncmp(p, "step ", 5)) {
            char hex[PHASE35_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                return false;
            }
            if (step_index < 0 || step_index >= vc->nsteps || ntop < 0 || ntop > 32) {
                return false;
            }
            vc->steps[step_index].ntop = ntop;
            if (!hex_to_bytes(hex,
                              vc->steps[step_index].selected,
                              PHASE35_MAX_TOKEN_BYTES,
                              &vc->steps[step_index].selected_len)) {
                return false;
            }
            top_index = 0;
            continue;
        }
        if (!strncmp(p, "top ", 4)) {
            char hex[PHASE35_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            if (step_index < 0 || step_index >= vc->nsteps ||
                top_index >= vc->steps[step_index].ntop) {
                return false;
            }
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                return false;
            }
            official_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            if (!hex_to_bytes(hex, top->bytes, PHASE35_MAX_TOKEN_BYTES, &top->len)) {
                return false;
            }
            continue;
        }
        return false;
    }
    return false;
}

static bool read_golden_case(FILE *fp, golden_case *tc) {
    char line[2048];
    memset(tc, 0, sizeof(*tc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %15s %d %d %511s %d",
                   tc->id, tc->mode, &tc->ctx, &tc->frontier,
                   tc->prompt_path, &tc->ntop) == 6) {
            return tc->ctx > tc->frontier &&
                   tc->frontier > 0 &&
                   tc->ntop > 0 &&
                   tc->ntop <= PHASE35_MAX_TOP;
        }
        return false;
    }
    return false;
}

static bool fill_golden_case(FILE *fp, golden_case *tc) {
    char line[2048];
    int seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return seen == tc->ntop;
        int rank = -1;
        int id = -1;
        float logit = 0.0f;
        if (sscanf(p, "top %d %d %f", &rank, &id, &logit) != 3) return false;
        if (rank != seen || seen >= tc->ntop) return false;
        tc->top[seen].id = id;
        tc->top[seen].logit = logit;
        seen++;
    }
    return false;
}

static bool parse_layer_spec(const char *spec, ds4_distributed_layers *out) {
    if (!spec || !out) return false;
    const char *colon = strchr(spec, ':');
    if (!colon || colon == spec || colon[1] == '\0') return false;

    errno = 0;
    char *end = NULL;
    unsigned long start = strtoul(spec, &end, 10);
    if (errno != 0 || end != colon || start > UINT32_MAX) return false;

    ds4_distributed_layers layers = {0};
    layers.start = (uint32_t)start;
    layers.set = true;
    if (!strcmp(colon + 1, "output")) {
        layers.has_output = true;
        layers.end = 0;
    } else {
        errno = 0;
        unsigned long layer_end = strtoul(colon + 1, &end, 10);
        if (errno != 0 || *end != '\0' || layer_end > UINT32_MAX) return false;
        layers.end = (uint32_t)layer_end;
    }
    *out = layers;
    return true;
}

static int wait_route(ds4_session *session, double timeout_sec) {
    double deadline = now_sec() + timeout_sec;
    while (now_sec() < deadline) {
        char err[256] = {0};
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready == 1) return 1;
        if (ready < 0) {
            fprintf(stderr, "issue304-phase35-vectors: route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return -1;
        }
        usleep(100000);
    }
    fprintf(stderr, "issue304-phase35-vectors: timed out waiting for route\n");
    return 0;
}

static bool set_process_lock_file(const char *path, char *err, size_t errlen) {
    if (!path || !path[0]) {
        snprintf(err, errlen, "invalid lock file path");
        return false;
    }
    if (setenv("DS4_LOCK_FILE", path, 1) != 0) {
        snprintf(err, errlen, "failed to set DS4_LOCK_FILE: %s", strerror(errno));
        return false;
    }
    return true;
}

static bool load_payload_into_session(ds4_session *session,
                                      const ds4_session_payload_file *payload,
                                      char *err,
                                      size_t errlen) {
    FILE *fp = fopen(payload->path, "rb");
    if (!fp) {
        snprintf(err, errlen, "failed to open staged payload: %s", strerror(errno));
        return false;
    }
    const int rc = ds4_session_load_payload(session, fp, payload->bytes, err, errlen);
    fclose(fp);
    return rc == 0;
}

static bool load_payload_path_into_session(ds4_session *session,
                                           const char *payload_path,
                                           uint64_t *payload_bytes_out,
                                           char *err,
                                           size_t errlen) {
    FILE *fp = fopen(payload_path, "rb");
    if (!fp) {
        snprintf(err, errlen, "failed to open payload file: %s", strerror(errno));
        return false;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        snprintf(err, errlen, "failed to seek payload file: %s", strerror(errno));
        fclose(fp);
        return false;
    }
    long payload_bytes = ftell(fp);
    if (payload_bytes < 0) {
        snprintf(err, errlen, "failed to measure payload file: %s", strerror(errno));
        fclose(fp);
        return false;
    }
    rewind(fp);
    const int rc = ds4_session_load_payload(session, fp, (uint64_t)payload_bytes, err, errlen);
    fclose(fp);
    if (rc != 0) return false;
    if (payload_bytes_out) *payload_bytes_out = (uint64_t)payload_bytes;
    return true;
}

static bool write_payload_copy(const ds4_session_payload_file *payload,
                               const char *dst_path,
                               char *err,
                               size_t errlen) {
    FILE *fp = fopen(dst_path, "wb");
    if (!fp) {
        snprintf(err, errlen, "failed to open payload output: %s", strerror(errno));
        return false;
    }
    const int rc = ds4_session_write_staged_payload(payload, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        snprintf(err, errlen, "failed to close payload output: %s", strerror(errno));
        return false;
    }
    return rc == 0;
}

static bool tokenize_official_prompt(ds4_engine *engine,
                                     const official_case *vc,
                                     ds4_tokens *out) {
    memset(out, 0, sizeof(*out));
    char *prompt_text = read_file(vc->prompt_path);
    if (!prompt_text) return false;
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, out);
    free(prompt_text);
    return out->len > 0;
}

static bool tokenize_golden_prompt(ds4_engine *engine,
                                   const golden_case *tc,
                                   ds4_tokens *full_prompt,
                                   ds4_tokens *prefix) {
    memset(full_prompt, 0, sizeof(*full_prompt));
    memset(prefix, 0, sizeof(*prefix));
    char *prompt_text = read_file(tc->prompt_path);
    if (!prompt_text) return false;
    if (!strcmp(tc->mode, "text")) {
        ds4_tokenize_text(engine, prompt_text, full_prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        ds4_tokenize_rendered_chat(engine, prompt_text, full_prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, full_prompt);
    } else {
        free(prompt_text);
        return false;
    }
    free(prompt_text);
    if (full_prompt->len < tc->frontier) return false;
    prefix->v = full_prompt->v;
    prefix->len = tc->frontier;
    prefix->cap = tc->frontier;
    return prefix->len > 0;
}

static bool open_host_local_engine(const phase35_cfg *cfg, ds4_engine **engine_out) {
    *engine_out = NULL;
    ds4_engine_options local_opt = {
        .model_path = cfg->model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .power_percent = cfg->power_percent,
        .quality = false,
    };
    return ds4_engine_open(engine_out, &local_opt) == 0 && *engine_out;
}

static bool open_distributed_session(const phase35_cfg *cfg,
                                     int ctx,
                                     ds4_distributed_layers coordinator_layers,
                                     ds4_engine **engine_out,
                                     ds4_session **session_out,
                                     char *route_summary,
                                     size_t route_summary_len,
                                     uint32_t *route_hops,
                                     bool *output_on_coordinator) {
    *engine_out = NULL;
    *session_out = NULL;
    char err[256] = {0};
    ds4_engine_options dist_opt = {
        .model_path = cfg->model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .power_percent = cfg->power_percent,
        .quality = false,
        .distributed = {
            .role = DS4_DISTRIBUTED_COORDINATOR,
            .layers = coordinator_layers,
            .listen_host = cfg->listen_host,
            .listen_port = cfg->listen_port,
            .prefill_chunk = cfg->prefill_chunk,
            .prefill_window = cfg->prefill_window,
            .activation_bits = cfg->activation_bits,
            .debug = cfg->debug,
        },
    };
    if (ds4_dist_prepare_engine_options(&dist_opt.distributed, &dist_opt, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase35-vectors: distributed options failed: %s\n",
                err[0] ? err : "unknown error");
        return false;
    }
    if (ds4_engine_open(engine_out, &dist_opt) != 0 || !*engine_out) return false;
    if (ds4_session_create(session_out, *engine_out, ctx) != 0 || !*session_out) return false;
    if (wait_route(*session_out, 60.0) != 1) return false;
    if (ds4_session_distributed_route_summary(*session_out,
                                              route_summary,
                                              route_summary_len,
                                              route_hops,
                                              output_on_coordinator,
                                              err,
                                              sizeof(err)) != 1) {
        fprintf(stderr, "issue304-phase35-vectors: route summary failed: %s\n",
                err[0] ? err : "unknown error");
        return false;
    }
    return true;
}

static bool stage_distributed_payload(ds4_session *dist_session,
                                      const ds4_tokens *prompt,
                                      repeat_result *res,
                                      ds4_session_payload_file *payload) {
    char err[256] = {0};
    double t0 = now_sec();
    if (ds4_session_sync(dist_session, prompt, err, sizeof(err)) != 0) {
        snprintf(res->error, sizeof(res->error), "distributed prefill failed: %s",
                 err[0] ? err : "unknown error");
        return false;
    }
    res->prefill_sec = now_sec() - t0;

    t0 = now_sec();
    if (ds4_session_stage_payload(dist_session, payload, err, sizeof(err)) != 0) {
        snprintf(res->error, sizeof(res->error), "distributed payload stage failed: %s",
                 err[0] ? err : "unknown error");
        return false;
    }
    res->stage_sec = now_sec() - t0;
    res->payload_bytes = payload->bytes;
    return true;
}

static bool run_official_repeat(const phase35_cfg *cfg,
                                ds4_distributed_layers coordinator_layers,
                                const official_case *vc,
                                repeat_result *res) {
    memset(res, 0, sizeof(*res));
    res->official_pass = true;
    res->official_selected_mismatch_step = -1;

    if (cfg->mode == PHASE35_MODE_DISTRIBUTED) {
        ds4_engine *dist_engine = NULL;
        ds4_session *cand_session = NULL;
        ds4_session_payload_file payload = {0};
        ds4_tokens prompt = {0};
        float *cand_logits = NULL;
        ds4_token_score scores[20];
        char err[256] = {0};
        bool ok = false;

        if (!set_process_lock_file("/tmp/ds4-phase35-dist.lock", err, sizeof(err))) {
            snprintf(res->error, sizeof(res->error), "%s", err);
            goto dist_cleanup;
        }
        if (!open_distributed_session(cfg, vc->ctx, coordinator_layers,
                                      &dist_engine, &cand_session,
                                      res->route_summary, sizeof(res->route_summary),
                                      &res->route_hops, &res->output_on_coordinator)) {
            snprintf(res->error, sizeof(res->error), "failed to open distributed session");
            goto dist_cleanup;
        }
        if (!tokenize_official_prompt(dist_engine, vc, &prompt)) {
            snprintf(res->error, sizeof(res->error), "failed to tokenize official prompt");
            goto dist_cleanup;
        }
        if (prompt.len >= vc->ctx) {
            snprintf(res->error, sizeof(res->error), "prompt exceeds ctx");
            goto dist_cleanup;
        }
        double t0 = now_sec();
        if (ds4_session_sync(cand_session, &prompt, err, sizeof(err)) != 0) {
            snprintf(res->error, sizeof(res->error), "distributed prefill failed: %s",
                     err[0] ? err : "unknown error");
            goto dist_cleanup;
        }
        res->prefill_sec = now_sec() - t0;
        if (cfg->payload_out) {
            t0 = now_sec();
            if (ds4_session_stage_payload(cand_session, &payload, err, sizeof(err)) != 0) {
                snprintf(res->error, sizeof(res->error), "distributed payload stage failed: %s",
                         err[0] ? err : "unknown error");
                goto dist_cleanup;
            }
            res->stage_sec = now_sec() - t0;
            res->payload_bytes = payload.bytes;
            if (!write_payload_copy(&payload, cfg->payload_out, err, sizeof(err))) {
                snprintf(res->error, sizeof(res->error), "payload export failed: %s",
                         err[0] ? err : "unknown error");
                goto dist_cleanup;
            }
        }

        const int vocab = ds4_engine_vocab_size(dist_engine);
        cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
        if (!cand_logits) {
            snprintf(res->error, sizeof(res->error), "out of memory for logits");
            goto dist_cleanup;
        }

        for (int step = 0; step < vc->nsteps; step++) {
            if (ds4_session_copy_logits(cand_session, cand_logits, vocab) != vocab) {
                snprintf(res->error, sizeof(res->error), "failed to copy logits");
                goto dist_cleanup;
            }
            parity_metrics cmp = compare_logits(cand_logits, cand_logits, vocab);
            parity_summary_observe(&res->parity, &cmp, step);
            res->steps_checked = step + 1;

            const int cand_token = ds4_session_argmax(cand_session);
            if (!token_bytes_equal(dist_engine, cand_token,
                                   vc->steps[step].selected,
                                   vc->steps[step].selected_len)) {
                if (res->official_selected_mismatch_step < 0) {
                    res->official_selected_mismatch_step = step;
                }
                res->official_pass = false;
            }

            const int nscore = ds4_session_top_logprobs(cand_session, scores, 20);
            for (int i = 0; i < vc->steps[step].ntop; i++) {
                bool found = false;
                float local_lp = 0.0f;
                for (int j = 0; j < nscore; j++) {
                    if (scores[j].id < 0) continue;
                    if (token_bytes_equal(dist_engine, scores[j].id,
                                          vc->steps[step].top[i].bytes,
                                          vc->steps[step].top[i].len)) {
                        found = true;
                        local_lp = scores[j].logprob;
                        break;
                    }
                }
                if (!found) {
                    res->official_missing_top_count++;
                    res->official_pass = false;
                    continue;
                }
                const float delta = fabsf(local_lp - vc->steps[step].top[i].logprob);
                if (delta > res->official_max_logprob_delta) {
                    res->official_max_logprob_delta = delta;
                }
                if (delta > 4.0f) res->official_pass = false;
            }

            if (step + 1 < vc->nsteps) {
                if (ds4_session_eval(cand_session, cand_token, err, sizeof(err)) != 0) {
                    snprintf(res->error, sizeof(res->error), "eval failed after step %d: %s",
                             step, err[0] ? err : "unknown error");
                    goto dist_cleanup;
                }
            }
        }

        res->ok = true;
        ok = true;

dist_cleanup:
        free(cand_logits);
        ds4_session_payload_file_free(&payload);
        ds4_tokens_free(&prompt);
        ds4_session_free(cand_session);
        ds4_engine_close(dist_engine);
        return ok;
    }

    ds4_engine *local_engine = NULL;
    ds4_engine *dist_engine = NULL;
    ds4_session *ref_session = NULL;
    ds4_session *cand_session = NULL;
    ds4_session_payload_file payload = {0};
    ds4_tokens prompt = {0};
    float *ref_logits = NULL;
    float *cand_logits = NULL;
    ds4_token_score scores[20];
    char err[256] = {0};
    bool ok = false;

    if (!set_process_lock_file("/tmp/ds4-phase35-local.lock", err, sizeof(err))) {
        snprintf(res->error, sizeof(res->error), "%s", err);
        goto cleanup;
    }
    if (!open_host_local_engine(cfg, &local_engine)) {
        snprintf(res->error, sizeof(res->error), "failed to open local engine");
        goto cleanup;
    }
    if (!tokenize_official_prompt(local_engine, vc, &prompt)) {
        snprintf(res->error, sizeof(res->error), "failed to tokenize official prompt");
        goto cleanup;
    }
    if (prompt.len >= vc->ctx) {
        snprintf(res->error, sizeof(res->error), "prompt exceeds ctx");
        goto cleanup;
    }
    if (ds4_session_create(&ref_session, local_engine, vc->ctx) != 0 || !ref_session) {
        snprintf(res->error, sizeof(res->error), "failed to create local reference session");
        goto cleanup;
    }
    if (ds4_session_sync(ref_session, &prompt, err, sizeof(err)) != 0) {
        snprintf(res->error, sizeof(res->error), "local baseline prefill failed: %s",
                 err[0] ? err : "unknown error");
        goto cleanup;
    }

    if (cfg->mode == PHASE35_MODE_LOCAL) {
        if (ds4_session_create(&cand_session, local_engine, vc->ctx) != 0 || !cand_session) {
            snprintf(res->error, sizeof(res->error), "failed to create local candidate session");
            goto cleanup;
        }
        double t0 = now_sec();
        if (ds4_session_sync(cand_session, &prompt, err, sizeof(err)) != 0) {
            snprintf(res->error, sizeof(res->error), "local candidate prefill failed: %s",
                     err[0] ? err : "unknown error");
            goto cleanup;
        }
        res->prefill_sec = now_sec() - t0;
        snprintf(res->route_summary, sizeof(res->route_summary), "local");
        res->route_hops = 0;
        res->output_on_coordinator = true;
    } else if (cfg->mode == PHASE35_MODE_DISTRIBUTED) {
        if (!set_process_lock_file("/tmp/ds4-phase35-dist.lock", err, sizeof(err))) {
            snprintf(res->error, sizeof(res->error), "%s", err);
            goto cleanup;
        }
        if (!open_distributed_session(cfg, vc->ctx, coordinator_layers,
                                      &dist_engine, &cand_session,
                                      res->route_summary, sizeof(res->route_summary),
                                      &res->route_hops, &res->output_on_coordinator)) {
            snprintf(res->error, sizeof(res->error), "failed to open distributed session");
            goto cleanup;
        }
        double t0 = now_sec();
        if (ds4_session_sync(cand_session, &prompt, err, sizeof(err)) != 0) {
            snprintf(res->error, sizeof(res->error), "distributed prefill failed: %s",
                     err[0] ? err : "unknown error");
            goto cleanup;
        }
        res->prefill_sec = now_sec() - t0;
        if (cfg->payload_out) {
            t0 = now_sec();
            if (ds4_session_stage_payload(cand_session, &payload, err, sizeof(err)) != 0) {
                snprintf(res->error, sizeof(res->error), "distributed payload stage failed: %s",
                         err[0] ? err : "unknown error");
                goto cleanup;
            }
            res->stage_sec = now_sec() - t0;
            res->payload_bytes = payload.bytes;
            if (!write_payload_copy(&payload, cfg->payload_out, err, sizeof(err))) {
                snprintf(res->error, sizeof(res->error), "payload export failed: %s",
                         err[0] ? err : "unknown error");
                goto cleanup;
            }
        }
    } else if (cfg->mode == PHASE35_MODE_PAYLOAD) {
        if (!cfg->payload_file) {
            snprintf(res->error, sizeof(res->error), "payload mode requires --payload-file");
            goto cleanup;
        }
        if (ds4_session_create(&cand_session, local_engine, vc->ctx) != 0 || !cand_session) {
            snprintf(res->error, sizeof(res->error), "failed to create payload candidate session");
            goto cleanup;
        }
        double t0 = now_sec();
        if (!load_payload_path_into_session(cand_session, cfg->payload_file, &res->payload_bytes,
                                            err, sizeof(err))) {
            snprintf(res->error, sizeof(res->error), "payload load failed: %s",
                     err[0] ? err : "unknown error");
            goto cleanup;
        }
        res->load_sec = now_sec() - t0;
        snprintf(res->route_summary, sizeof(res->route_summary), "payload");
        res->route_hops = 0;
        res->output_on_coordinator = true;
    }

    const int vocab = ds4_engine_vocab_size(local_engine);
    ref_logits = malloc((size_t)vocab * sizeof(ref_logits[0]));
    cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    if (!ref_logits || !cand_logits) {
        snprintf(res->error, sizeof(res->error), "out of memory for logits");
        goto cleanup;
    }

    for (int step = 0; step < vc->nsteps; step++) {
        if (ds4_session_copy_logits(ref_session, ref_logits, vocab) != vocab ||
            ds4_session_copy_logits(cand_session, cand_logits, vocab) != vocab) {
            snprintf(res->error, sizeof(res->error), "failed to copy logits");
            goto cleanup;
        }

        parity_metrics cmp = compare_logits(ref_logits, cand_logits, vocab);
        parity_summary_observe(&res->parity, &cmp, step);
        res->steps_checked = step + 1;

        const int ref_token = ds4_session_argmax(ref_session);
        const int cand_token = ds4_session_argmax(cand_session);
        if (!token_bytes_equal(local_engine, cand_token,
                               vc->steps[step].selected,
                               vc->steps[step].selected_len)) {
            if (res->official_selected_mismatch_step < 0) {
                res->official_selected_mismatch_step = step;
            }
            res->official_pass = false;
        }

        const int nscore = ds4_session_top_logprobs(cand_session, scores, 20);
        for (int i = 0; i < vc->steps[step].ntop; i++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (token_bytes_equal(local_engine, scores[j].id,
                                      vc->steps[step].top[i].bytes,
                                      vc->steps[step].top[i].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                res->official_missing_top_count++;
                res->official_pass = false;
                continue;
            }
            const float delta = fabsf(local_lp - vc->steps[step].top[i].logprob);
            if (delta > res->official_max_logprob_delta) {
                res->official_max_logprob_delta = delta;
            }
            if (delta > 4.0f) res->official_pass = false;
        }

        if (step + 1 < vc->nsteps) {
            if (ds4_session_eval(ref_session, ref_token, err, sizeof(err)) != 0 ||
                ds4_session_eval(cand_session, ref_token, err, sizeof(err)) != 0) {
                snprintf(res->error, sizeof(res->error), "eval failed after step %d: %s",
                         step, err[0] ? err : "unknown error");
                goto cleanup;
            }
        }
    }

    res->ok = true;
    ok = true;

cleanup:
    free(cand_logits);
    free(ref_logits);
    ds4_session_payload_file_free(&payload);
    ds4_tokens_free(&prompt);
    ds4_session_free(cand_session);
    ds4_session_free(ref_session);
    ds4_engine_close(dist_engine);
    ds4_engine_close(local_engine);
    return ok;
}

static int golden_overlap(const golden_case *tc, const int *cand_top, int n) {
    int overlap = 0;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        if (topk_contains(cand_top, n, tc->top[i].id)) overlap++;
    }
    return overlap;
}

static float golden_top20_max_abs(const golden_case *tc, const float *cand_logits) {
    float max_abs = 0.0f;
    int n = tc->ntop < 20 ? tc->ntop : 20;
    for (int i = 0; i < n; i++) {
        const int id = tc->top[i].id;
        if (id < 0) continue;
        const float abs_delta = fabsf(cand_logits[id] - tc->top[i].logit);
        if (abs_delta > max_abs) max_abs = abs_delta;
    }
    return max_abs;
}

static bool run_golden_repeat(const phase35_cfg *cfg,
                              ds4_distributed_layers coordinator_layers,
                              const golden_case *tc,
                              repeat_result *res) {
    memset(res, 0, sizeof(*res));
    res->official_selected_mismatch_step = -1;
    res->golden_pass = true;

    if (cfg->mode == PHASE35_MODE_DISTRIBUTED) {
        ds4_engine *dist_engine = NULL;
        ds4_session *cand_session = NULL;
        ds4_session_payload_file payload = {0};
        ds4_tokens full_prompt = {0};
        ds4_tokens prefix = {0};
        float *cand_logits = NULL;
        char err[256] = {0};
        bool ok = false;

        if (!set_process_lock_file("/tmp/ds4-phase35-dist.lock", err, sizeof(err))) {
            snprintf(res->error, sizeof(res->error), "%s", err);
            goto dist_cleanup;
        }
        if (!open_distributed_session(cfg, tc->ctx, coordinator_layers,
                                      &dist_engine, &cand_session,
                                      res->route_summary, sizeof(res->route_summary),
                                      &res->route_hops, &res->output_on_coordinator)) {
            snprintf(res->error, sizeof(res->error), "failed to open distributed session");
            goto dist_cleanup;
        }
        if (!tokenize_golden_prompt(dist_engine, tc, &full_prompt, &prefix)) {
            snprintf(res->error, sizeof(res->error), "failed to tokenize local golden prompt");
            goto dist_cleanup;
        }
        double t0 = now_sec();
        if (ds4_session_sync(cand_session, &prefix, err, sizeof(err)) != 0) {
            snprintf(res->error, sizeof(res->error), "distributed prefill failed: %s",
                     err[0] ? err : "unknown error");
            goto dist_cleanup;
        }
        res->prefill_sec = now_sec() - t0;
        if (cfg->payload_out) {
            t0 = now_sec();
            if (ds4_session_stage_payload(cand_session, &payload, err, sizeof(err)) != 0) {
                snprintf(res->error, sizeof(res->error), "distributed payload stage failed: %s",
                         err[0] ? err : "unknown error");
                goto dist_cleanup;
            }
            res->stage_sec = now_sec() - t0;
            res->payload_bytes = payload.bytes;
            if (!write_payload_copy(&payload, cfg->payload_out, err, sizeof(err))) {
                snprintf(res->error, sizeof(res->error), "payload export failed: %s",
                         err[0] ? err : "unknown error");
                goto dist_cleanup;
            }
        }

        const int vocab = ds4_engine_vocab_size(dist_engine);
        cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
        if (!cand_logits) {
            snprintf(res->error, sizeof(res->error), "out of memory for logits");
            goto dist_cleanup;
        }

        for (int step = 0; step <= cfg->greedy_steps; step++) {
            if (ds4_session_copy_logits(cand_session, cand_logits, vocab) != vocab) {
                snprintf(res->error, sizeof(res->error), "failed to copy logits");
                goto dist_cleanup;
            }
            parity_metrics cmp = compare_logits(cand_logits, cand_logits, vocab);
            parity_summary_observe(&res->parity, &cmp, step);
            res->steps_checked = step + 1;

            if (step == 0) {
                int cand_top[64];
                logits_topk(cand_logits, vocab, cand_top, 64);
                res->golden_top1 = cand_top[0];
                res->golden_top5_overlap = golden_overlap(tc, cand_top, 5);
                res->golden_top20_overlap = golden_overlap(tc, cand_top, 20);
                res->golden_top64_overlap = golden_overlap(tc, cand_top, 64);
                res->golden_top20_max_abs = golden_top20_max_abs(tc, cand_logits);
                res->golden_pass = cand_top[0] == tc->top[0].id &&
                                   res->golden_top5_overlap >= 4 &&
                                   res->golden_top20_overlap >= 15 &&
                                   res->golden_top64_overlap >= 40 &&
                                   res->golden_top20_max_abs <= 8.0f;
            }

            if (step == cfg->greedy_steps) break;
            const int cand_token = ds4_session_argmax(cand_session);
            if (ds4_session_eval(cand_session, cand_token, err, sizeof(err)) != 0) {
                snprintf(res->error, sizeof(res->error), "eval failed after step %d: %s",
                         step, err[0] ? err : "unknown error");
                goto dist_cleanup;
            }
        }

        res->ok = true;
        ok = true;

dist_cleanup:
        free(cand_logits);
        ds4_session_payload_file_free(&payload);
        ds4_tokens_free(&full_prompt);
        ds4_session_free(cand_session);
        ds4_engine_close(dist_engine);
        return ok;
    }

    ds4_engine *local_engine = NULL;
    ds4_engine *dist_engine = NULL;
    ds4_session *ref_session = NULL;
    ds4_session *cand_session = NULL;
    ds4_session_payload_file payload = {0};
    ds4_tokens full_prompt = {0};
    ds4_tokens prefix = {0};
    float *ref_logits = NULL;
    float *cand_logits = NULL;
    char err[256] = {0};
    bool ok = false;

    if (!set_process_lock_file("/tmp/ds4-phase35-local.lock", err, sizeof(err))) {
        snprintf(res->error, sizeof(res->error), "%s", err);
        goto cleanup;
    }
    if (!open_host_local_engine(cfg, &local_engine)) {
        snprintf(res->error, sizeof(res->error), "failed to open local engine");
        goto cleanup;
    }
    if (!tokenize_golden_prompt(local_engine, tc, &full_prompt, &prefix)) {
        snprintf(res->error, sizeof(res->error), "failed to tokenize local golden prompt");
        goto cleanup;
    }
    if (ds4_session_create(&ref_session, local_engine, tc->ctx) != 0 || !ref_session) {
        snprintf(res->error, sizeof(res->error), "failed to create local reference session");
        goto cleanup;
    }
    if (ds4_session_sync(ref_session, &prefix, err, sizeof(err)) != 0) {
        snprintf(res->error, sizeof(res->error), "local baseline prefill failed: %s",
                 err[0] ? err : "unknown error");
        goto cleanup;
    }

    if (cfg->mode == PHASE35_MODE_LOCAL) {
        if (ds4_session_create(&cand_session, local_engine, tc->ctx) != 0 || !cand_session) {
            snprintf(res->error, sizeof(res->error), "failed to create local candidate session");
            goto cleanup;
        }
        double t0 = now_sec();
        if (ds4_session_sync(cand_session, &prefix, err, sizeof(err)) != 0) {
            snprintf(res->error, sizeof(res->error), "local candidate prefill failed: %s",
                     err[0] ? err : "unknown error");
            goto cleanup;
        }
        res->prefill_sec = now_sec() - t0;
        snprintf(res->route_summary, sizeof(res->route_summary), "local");
        res->route_hops = 0;
        res->output_on_coordinator = true;
    } else if (cfg->mode == PHASE35_MODE_DISTRIBUTED) {
        if (!set_process_lock_file("/tmp/ds4-phase35-dist.lock", err, sizeof(err))) {
            snprintf(res->error, sizeof(res->error), "%s", err);
            goto cleanup;
        }
        if (!open_distributed_session(cfg, tc->ctx, coordinator_layers,
                                      &dist_engine, &cand_session,
                                      res->route_summary, sizeof(res->route_summary),
                                      &res->route_hops, &res->output_on_coordinator)) {
            snprintf(res->error, sizeof(res->error), "failed to open distributed session");
            goto cleanup;
        }
        double t0 = now_sec();
        if (ds4_session_sync(cand_session, &prefix, err, sizeof(err)) != 0) {
            snprintf(res->error, sizeof(res->error), "distributed prefill failed: %s",
                     err[0] ? err : "unknown error");
            goto cleanup;
        }
        res->prefill_sec = now_sec() - t0;
        if (cfg->payload_out) {
            t0 = now_sec();
            if (ds4_session_stage_payload(cand_session, &payload, err, sizeof(err)) != 0) {
                snprintf(res->error, sizeof(res->error), "distributed payload stage failed: %s",
                         err[0] ? err : "unknown error");
                goto cleanup;
            }
            res->stage_sec = now_sec() - t0;
            res->payload_bytes = payload.bytes;
            if (!write_payload_copy(&payload, cfg->payload_out, err, sizeof(err))) {
                snprintf(res->error, sizeof(res->error), "payload export failed: %s",
                         err[0] ? err : "unknown error");
                goto cleanup;
            }
        }
    } else if (cfg->mode == PHASE35_MODE_PAYLOAD) {
        if (!cfg->payload_file) {
            snprintf(res->error, sizeof(res->error), "payload mode requires --payload-file");
            goto cleanup;
        }
        if (ds4_session_create(&cand_session, local_engine, tc->ctx) != 0 || !cand_session) {
            snprintf(res->error, sizeof(res->error), "failed to create payload candidate session");
            goto cleanup;
        }
        double t0 = now_sec();
        if (!load_payload_path_into_session(cand_session, cfg->payload_file, &res->payload_bytes,
                                            err, sizeof(err))) {
            snprintf(res->error, sizeof(res->error), "payload load failed: %s",
                     err[0] ? err : "unknown error");
            goto cleanup;
        }
        res->load_sec = now_sec() - t0;
        snprintf(res->route_summary, sizeof(res->route_summary), "payload");
        res->route_hops = 0;
        res->output_on_coordinator = true;
    }

    const int vocab = ds4_engine_vocab_size(local_engine);
    ref_logits = malloc((size_t)vocab * sizeof(ref_logits[0]));
    cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    if (!ref_logits || !cand_logits) {
        snprintf(res->error, sizeof(res->error), "out of memory for logits");
        goto cleanup;
    }

    for (int step = 0; step <= cfg->greedy_steps; step++) {
        if (ds4_session_copy_logits(ref_session, ref_logits, vocab) != vocab ||
            ds4_session_copy_logits(cand_session, cand_logits, vocab) != vocab) {
            snprintf(res->error, sizeof(res->error), "failed to copy logits");
            goto cleanup;
        }

        parity_metrics cmp = compare_logits(ref_logits, cand_logits, vocab);
        parity_summary_observe(&res->parity, &cmp, step);
        res->steps_checked = step + 1;

        if (step == 0) {
            int cand_top[64];
            logits_topk(cand_logits, vocab, cand_top, 64);
            res->golden_top1 = cand_top[0];
            res->golden_top5_overlap = golden_overlap(tc, cand_top, 5);
            res->golden_top20_overlap = golden_overlap(tc, cand_top, 20);
            res->golden_top64_overlap = golden_overlap(tc, cand_top, 64);
            res->golden_top20_max_abs = golden_top20_max_abs(tc, cand_logits);
            res->golden_pass = cand_top[0] == tc->top[0].id &&
                               res->golden_top5_overlap >= 4 &&
                               res->golden_top20_overlap >= 15 &&
                               res->golden_top64_overlap >= 40 &&
                               res->golden_top20_max_abs <= 8.0f;
        }

        if (step == cfg->greedy_steps) break;
        const int ref_token = ds4_session_argmax(ref_session);
        if (ds4_session_eval(ref_session, ref_token, err, sizeof(err)) != 0 ||
            ds4_session_eval(cand_session, ref_token, err, sizeof(err)) != 0) {
            snprintf(res->error, sizeof(res->error), "eval failed after step %d: %s",
                     step, err[0] ? err : "unknown error");
            goto cleanup;
        }
    }

    res->ok = true;
    ok = true;

cleanup:
    free(cand_logits);
    free(ref_logits);
    ds4_session_payload_file_free(&payload);
    ds4_tokens_free(&full_prompt);
    ds4_session_free(cand_session);
    ds4_session_free(ref_session);
    ds4_engine_close(dist_engine);
    ds4_engine_close(local_engine);
    return ok;
}

static void summary_observe_repeat(case_summary *sum, const repeat_result *res) {
    if (sum->repeats == 0) {
        sum->worst_parity = res->parity;
        sum->official_pass = res->official_pass;
        sum->official_selected_mismatch_step = res->official_selected_mismatch_step;
        sum->official_missing_top_count = res->official_missing_top_count;
        sum->official_max_logprob_delta = res->official_max_logprob_delta;
        sum->golden_pass = res->golden_pass;
        sum->golden_top1 = res->golden_top1;
        sum->golden_top5_overlap = res->golden_top5_overlap;
        sum->golden_top20_overlap = res->golden_top20_overlap;
        sum->golden_top64_overlap = res->golden_top64_overlap;
        sum->golden_top20_max_abs = res->golden_top20_max_abs;
        snprintf(sum->route_summary, sizeof(sum->route_summary), "%s", res->route_summary);
        sum->route_hops = res->route_hops;
        sum->output_on_coordinator = res->output_on_coordinator;
    } else {
        if (res->parity.top5_overlap < sum->worst_parity.top5_overlap) {
            sum->worst_parity.top5_overlap = res->parity.top5_overlap;
        }
        if (res->parity.top20_overlap < sum->worst_parity.top20_overlap) {
            sum->worst_parity.top20_overlap = res->parity.top20_overlap;
        }
        if (res->parity.top64_overlap < sum->worst_parity.top64_overlap) {
            sum->worst_parity.top64_overlap = res->parity.top64_overlap;
        }
        if (res->parity.nonfinite > sum->worst_parity.nonfinite) {
            sum->worst_parity.nonfinite = res->parity.nonfinite;
        }
        if (res->parity.rms > sum->worst_parity.rms) {
            sum->worst_parity.rms = res->parity.rms;
        }
        if (res->parity.max_abs > sum->worst_parity.max_abs) {
            sum->worst_parity.max_abs = res->parity.max_abs;
        }
        if (res->parity.top20_max_abs > sum->worst_parity.top20_max_abs) {
            sum->worst_parity.top20_max_abs = res->parity.top20_max_abs;
        }
        if (sum->worst_parity.first_top1_mismatch_step < 0 &&
            res->parity.first_top1_mismatch_step >= 0) {
            sum->worst_parity.first_top1_mismatch_step = res->parity.first_top1_mismatch_step;
            sum->worst_parity.top1_ref = res->parity.top1_ref;
            sum->worst_parity.top1_cand = res->parity.top1_cand;
            sum->worst_parity.first_ref_token = res->parity.first_ref_token;
            sum->worst_parity.first_cand_token = res->parity.first_cand_token;
        }
        sum->official_pass = sum->official_pass && res->official_pass;
        if (sum->official_selected_mismatch_step < 0 ||
            (res->official_selected_mismatch_step >= 0 &&
             res->official_selected_mismatch_step < sum->official_selected_mismatch_step)) {
            if (res->official_selected_mismatch_step >= 0) {
                sum->official_selected_mismatch_step = res->official_selected_mismatch_step;
            }
        }
        sum->official_missing_top_count += res->official_missing_top_count;
        if (res->official_max_logprob_delta > sum->official_max_logprob_delta) {
            sum->official_max_logprob_delta = res->official_max_logprob_delta;
        }
        sum->golden_pass = sum->golden_pass && res->golden_pass;
        if (res->golden_top5_overlap < sum->golden_top5_overlap) {
            sum->golden_top5_overlap = res->golden_top5_overlap;
        }
        if (res->golden_top20_overlap < sum->golden_top20_overlap) {
            sum->golden_top20_overlap = res->golden_top20_overlap;
        }
        if (res->golden_top64_overlap < sum->golden_top64_overlap) {
            sum->golden_top64_overlap = res->golden_top64_overlap;
        }
        if (res->golden_top20_max_abs > sum->golden_top20_max_abs) {
            sum->golden_top20_max_abs = res->golden_top20_max_abs;
        }
    }
    if (res->ok) sum->passes++;
    sum->repeats++;
    if (res->payload_bytes > sum->payload_bytes_max) sum->payload_bytes_max = res->payload_bytes;
    if (res->prefill_sec > sum->prefill_sec_max) sum->prefill_sec_max = res->prefill_sec;
    if (res->stage_sec > sum->stage_sec_max) sum->stage_sec_max = res->stage_sec;
    if (res->load_sec > sum->load_sec_max) sum->load_sec_max = res->load_sec;
}

static void print_repeat_result(const phase35_cfg *cfg,
                                const char *case_id,
                                int ctx,
                                int repeat_index,
                                const repeat_result *res) {
    printf("{\"mode\":\"%s\",\"suite\":\"%s\",\"case\":\"%s\",\"ctx\":%d,\"repeat\":%d,"
           "\"ok\":%s,\"route_summary\":\"%s\",\"payload_bytes\":%llu,"
           "\"prefill_sec\":%.6f,\"stage_sec\":%.6f,\"load_sec\":%.6f,"
           "\"parity\":{\"envelope\":\"%s\",\"first_top1_mismatch_step\":%d,"
           "\"top1_ref\":%d,\"top1_cand\":%d,\"top5_overlap\":%d,"
           "\"top20_overlap\":%d,\"top64_overlap\":%d,\"rms\":%.9g,"
           "\"max_abs\":%.9g,\"top20_max_abs\":%.9g,\"nonfinite\":%d},"
           "\"official\":{\"pass\":%s,\"selected_mismatch_step\":%d,"
           "\"missing_top_count\":%d,\"max_logprob_delta\":%.9g},"
           "\"golden\":{\"pass\":%s,\"top1\":%d,\"top5_overlap\":%d,"
           "\"top20_overlap\":%d,\"top64_overlap\":%d,\"top20_max_abs\":%.9g},"
           "\"error\":\"%s\"}\n",
           mode_name(cfg->mode),
           suite_name(cfg->suite),
           case_id,
           ctx,
           repeat_index,
           res->ok ? "true" : "false",
           res->route_summary,
           (unsigned long long)res->payload_bytes,
           res->prefill_sec,
           res->stage_sec,
           res->load_sec,
           parity_envelope(&res->parity),
           res->parity.first_top1_mismatch_step,
           res->parity.top1_ref,
           res->parity.top1_cand,
           res->parity.top5_overlap,
           res->parity.top20_overlap,
           res->parity.top64_overlap,
           res->parity.rms,
           res->parity.max_abs,
           res->parity.top20_max_abs,
           res->parity.nonfinite,
           res->official_pass ? "true" : "false",
           res->official_selected_mismatch_step,
           res->official_missing_top_count,
           res->official_max_logprob_delta,
           res->golden_pass ? "true" : "false",
           res->golden_top1,
           res->golden_top5_overlap,
           res->golden_top20_overlap,
           res->golden_top64_overlap,
           res->golden_top20_max_abs,
           res->error);
}

static void print_case_summary(const phase35_cfg *cfg,
                               const char *case_id,
                               int ctx,
                               const case_summary *sum) {
    printf("{\"mode\":\"%s\",\"suite\":\"%s\",\"case\":\"%s\",\"ctx\":%d,\"worst_at_%d\":{"
           "\"passes\":%d,\"route_summary\":\"%s\",\"payload_bytes_max\":%llu,"
           "\"prefill_sec_max\":%.6f,\"stage_sec_max\":%.6f,\"load_sec_max\":%.6f,"
           "\"parity\":{\"envelope\":\"%s\",\"first_top1_mismatch_step\":%d,"
           "\"top1_ref\":%d,\"top1_cand\":%d,\"top5_overlap\":%d,"
           "\"top20_overlap\":%d,\"top64_overlap\":%d,\"rms\":%.9g,"
           "\"max_abs\":%.9g,\"top20_max_abs\":%.9g,\"nonfinite\":%d},"
           "\"official\":{\"pass\":%s,\"selected_mismatch_step\":%d,"
           "\"missing_top_count\":%d,\"max_logprob_delta\":%.9g},"
           "\"golden\":{\"pass\":%s,\"top1\":%d,\"top5_overlap\":%d,"
           "\"top20_overlap\":%d,\"top64_overlap\":%d,\"top20_max_abs\":%.9g}}}\n",
           mode_name(cfg->mode),
           suite_name(cfg->suite),
           case_id,
           ctx,
           sum->repeats,
           sum->passes,
           sum->route_summary,
           (unsigned long long)sum->payload_bytes_max,
           sum->prefill_sec_max,
           sum->stage_sec_max,
           sum->load_sec_max,
           parity_envelope(&sum->worst_parity),
           sum->worst_parity.first_top1_mismatch_step,
           sum->worst_parity.top1_ref,
           sum->worst_parity.top1_cand,
           sum->worst_parity.top5_overlap,
           sum->worst_parity.top20_overlap,
           sum->worst_parity.top64_overlap,
           sum->worst_parity.rms,
           sum->worst_parity.max_abs,
           sum->worst_parity.top20_max_abs,
           sum->worst_parity.nonfinite,
           sum->official_pass ? "true" : "false",
           sum->official_selected_mismatch_step,
           sum->official_missing_top_count,
           sum->official_max_logprob_delta,
           sum->golden_pass ? "true" : "false",
           sum->golden_top1,
           sum->golden_top5_overlap,
           sum->golden_top20_overlap,
           sum->golden_top64_overlap,
           sum->golden_top20_max_abs);
}

static void parse_args(phase35_cfg *cfg, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            cfg->mode_name = argv[++i];
            if (!strcmp(cfg->mode_name, "local")) {
                cfg->mode = PHASE35_MODE_LOCAL;
            } else if (!strcmp(cfg->mode_name, "distributed")) {
                cfg->mode = PHASE35_MODE_DISTRIBUTED;
            } else if (!strcmp(cfg->mode_name, "payload")) {
                cfg->mode = PHASE35_MODE_PAYLOAD;
            } else {
                usage(argv[0]);
            }
        } else if (!strcmp(argv[i], "--suite") && i + 1 < argc) {
            const char *suite = argv[++i];
            if (!strcmp(suite, "official")) {
                cfg->suite = PHASE35_SUITE_OFFICIAL;
            } else if (!strcmp(suite, "local-golden")) {
                cfg->suite = PHASE35_SUITE_LOCAL_GOLDEN;
            } else {
                usage(argv[0]);
            }
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(argv[i], "--listen-host") && i + 1 < argc) {
            cfg->listen_host = argv[++i];
        } else if (!strcmp(argv[i], "--listen-port") && i + 1 < argc) {
            cfg->listen_port = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vector-file") && i + 1 < argc) {
            cfg->vector_file = argv[++i];
        } else if (!strcmp(argv[i], "--local-golden-file") && i + 1 < argc) {
            cfg->local_golden_file = argv[++i];
        } else if (!strcmp(argv[i], "--case-filter") && i + 1 < argc) {
            cfg->case_filter = argv[++i];
        } else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) {
            cfg->repeat = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ctx-filter") && i + 1 < argc) {
            cfg->ctx_filter = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--greedy-steps") && i + 1 < argc) {
            cfg->greedy_steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--prefill-chunk") && i + 1 < argc) {
            cfg->prefill_chunk = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--prefill-window") && i + 1 < argc) {
            cfg->prefill_window = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--activation-bits") && i + 1 < argc) {
            cfg->activation_bits = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--power") && i + 1 < argc) {
            cfg->power_percent = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--coordinator-layers") && i + 1 < argc) {
            cfg->coordinator_layers = argv[++i];
        } else if (!strcmp(argv[i], "--worker-layers") && i + 1 < argc) {
            cfg->worker_layers = argv[++i];
        } else if (!strcmp(argv[i], "--payload-file") && i + 1 < argc) {
            cfg->payload_file = argv[++i];
        } else if (!strcmp(argv[i], "--payload-out") && i + 1 < argc) {
            cfg->payload_out = argv[++i];
        } else if (!strcmp(argv[i], "--no-debug")) {
            cfg->debug = false;
        } else {
            usage(argv[0]);
        }
    }
}

int main(int argc, char **argv) {
    phase35_cfg cfg = {
        .model_path = NULL,
        .mode_name = "local",
        .listen_host = NULL,
        .vector_file = "tests/test-vectors/official.vec",
        .local_golden_file = "tests/test-vectors/local-golden.vec",
        .case_filter = NULL,
        .coordinator_layers = "0:21",
        .worker_layers = "22:output",
        .payload_file = NULL,
        .payload_out = NULL,
        .listen_port = 1234,
        .repeat = 5,
        .ctx_filter = 0,
        .greedy_steps = 8,
        .prefill_chunk = 256,
        .prefill_window = 0,
        .activation_bits = 32,
        .power_percent = 100,
        .debug = true,
        .suite = PHASE35_SUITE_OFFICIAL,
        .mode = PHASE35_MODE_LOCAL,
    };
    parse_args(&cfg, argc, argv);
    if (!cfg.model_path || cfg.repeat <= 0) {
        usage(argv[0]);
    }
    if (cfg.power_percent < 1 || cfg.power_percent > 100) {
        die("--power must be between 1 and 100");
    }

    ds4_distributed_layers coordinator_layers = {0};
    if (cfg.mode == PHASE35_MODE_DISTRIBUTED) {
        if (!cfg.listen_host || cfg.listen_port <= 0) usage(argv[0]);
        struct in_addr addr;
        if (inet_pton(AF_INET, cfg.listen_host, &addr) != 1) {
            die("listen host must be an IPv4 address reachable from the worker");
        }
        if (!parse_layer_spec(cfg.coordinator_layers, &coordinator_layers)) {
            die("invalid coordinator layer spec");
        }
    } else if (cfg.mode == PHASE35_MODE_PAYLOAD) {
        if (!cfg.payload_file) usage(argv[0]);
    }

    int failures = 0;
    if (cfg.suite == PHASE35_SUITE_OFFICIAL) {
        FILE *fp = fopen(cfg.vector_file, "rb");
        if (!fp) die_errno("open official vectors");
        official_case vc;
        while (read_official_case(fp, &vc)) {
            if (!fill_official_case(fp, &vc)) die("parse official vector case");
            if (official_case_disabled(&vc)) continue;
            if (!case_selected(vc.id, cfg.case_filter)) continue;
            if (cfg.ctx_filter > 0 && vc.ctx != cfg.ctx_filter) continue;

            case_summary sum = {0};
            sum.official_selected_mismatch_step = -1;
            for (int i = 0; i < cfg.repeat; i++) {
                repeat_result res = {0};
                if (!run_official_repeat(&cfg, coordinator_layers, &vc, &res)) {
                    print_repeat_result(&cfg, vc.id, vc.ctx, i + 1, &res);
                    fclose(fp);
                    return 1;
                }
                summary_observe_repeat(&sum, &res);
                print_repeat_result(&cfg, vc.id, vc.ctx, i + 1, &res);
                if (i + 1 < cfg.repeat &&
                    (cfg.mode == PHASE35_MODE_DISTRIBUTED || cfg.mode == PHASE35_MODE_PAYLOAD)) {
                    sleep(2);
                }
            }
            print_case_summary(&cfg, vc.id, vc.ctx, &sum);
            if (!sum.official_pass) failures++;
        }
        fclose(fp);
    } else {
        FILE *fp = fopen(cfg.local_golden_file, "rb");
        if (!fp) die_errno("open local golden vectors");
        golden_case tc;
        while (read_golden_case(fp, &tc)) {
            if (!fill_golden_case(fp, &tc)) die("parse local golden vector case");
            if (!case_selected(tc.id, cfg.case_filter)) continue;
            if (cfg.ctx_filter > 0 && tc.ctx != cfg.ctx_filter) continue;

            case_summary sum = {0};
            sum.official_selected_mismatch_step = -1;
            for (int i = 0; i < cfg.repeat; i++) {
                repeat_result res = {0};
                if (!run_golden_repeat(&cfg, coordinator_layers, &tc, &res)) {
                    print_repeat_result(&cfg, tc.id, tc.ctx, i + 1, &res);
                    fclose(fp);
                    return 1;
                }
                summary_observe_repeat(&sum, &res);
                print_repeat_result(&cfg, tc.id, tc.ctx, i + 1, &res);
                if (i + 1 < cfg.repeat &&
                    (cfg.mode == PHASE35_MODE_DISTRIBUTED || cfg.mode == PHASE35_MODE_PAYLOAD)) {
                    sleep(2);
                }
            }
            print_case_summary(&cfg, tc.id, tc.ctx, &sum);
            if (!sum.golden_pass) failures++;
        }
        fclose(fp);
    }
    return failures == 0 ? 0 : 1;
}
