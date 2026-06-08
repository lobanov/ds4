#include "../ds4.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *model_path;
    const char *prompt_file;
    const char *system_prompt;
    const char *followup_prompt;
    int ctx_size;
    int gen_tokens;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool normal_eval;
} cfg_t;

typedef struct {
    int top1_ref;
    int top1_cand;
    int top5_overlap;
    int top10_overlap;
    int top20_overlap;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
} parity_metrics;

typedef struct {
    int *tokens;
    int token_count;
} token_run;

static void die_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model FILE [--prompt-file FILE] [--system TEXT] "
            "[--followup TEXT] [--ctx N] [--gen-tokens N] [--temp F] "
            "[--top-p F] [--min-p F] [--seed N] [--normal-eval]\n",
            argv0);
    exit(2);
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

static void parse_args(cfg_t *cfg, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            cfg->prompt_file = argv[++i];
        } else if (!strcmp(argv[i], "--system") && i + 1 < argc) {
            cfg->system_prompt = argv[++i];
        } else if (!strcmp(argv[i], "--followup") && i + 1 < argc) {
            cfg->followup_prompt = argv[++i];
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            cfg->ctx_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--gen-tokens") && i + 1 < argc) {
            cfg->gen_tokens = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--temp") && i + 1 < argc) {
            cfg->temperature = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) {
            cfg->top_p = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) {
            cfg->min_p = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            cfg->seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--normal-eval")) {
            cfg->normal_eval = true;
        } else {
            die_usage(argv[0]);
        }
    }
    if (!cfg->model_path) die_usage(argv[0]);
}

static bool run_tokens(ds4_engine *engine,
                       ds4_session *session,
                       const cfg_t *cfg,
                       token_run *out) {
    char err[256] = {0};
    memset(out, 0, sizeof(*out));
    out->tokens = calloc((size_t)cfg->gen_tokens, sizeof(out->tokens[0]));
    if (!out->tokens) return false;

    if (!cfg->normal_eval) {
        for (int i = 0; i < cfg->gen_tokens; i++) {
            const int tok = ds4_session_argmax(session);
            out->tokens[i] = tok;
            out->token_count = i + 1;
            if (tok == ds4_token_eos(engine)) break;
            if (ds4_session_eval(session, tok, err, sizeof(err)) != 0) {
                fprintf(stderr, "issue304-phase55-prefill-vs-decode: eval failed: %s\n",
                        err[0] ? err : "unknown error");
                return false;
            }
        }
        return true;
    }

    uint64_t rng = cfg->seed;
    for (int i = 0; i < cfg->gen_tokens; i++) {
        const int tok = ds4_session_sample(session,
                                           cfg->temperature,
                                           0,
                                           cfg->top_p,
                                           cfg->min_p,
                                           &rng);
        out->tokens[i] = tok;
        out->token_count = i + 1;
        if (tok == ds4_token_eos(engine)) break;
        if (ds4_session_eval(session, tok, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase55-prefill-vs-decode: eval failed: %s\n",
                    err[0] ? err : "unknown error");
            return false;
        }
    }
    return true;
}

static void token_run_free(token_run *run) {
    free(run->tokens);
    memset(run, 0, sizeof(*run));
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
    for (int i = 0; i < k; i++) if (top[i] == id) return true;
    return false;
}

static parity_metrics compare_logits(const float *ref, const float *cand, int vocab) {
    int ref_top[64];
    int cand_top[64];
    logits_topk(ref, vocab, ref_top, 64);
    logits_topk(cand, vocab, cand_top, 64);

    int top5_overlap = 0;
    int top10_overlap = 0;
    int top20_overlap = 0;
    for (int i = 0; i < 20; i++) {
        if (ref_top[i] < 0) continue;
        if (i < 5 && topk_contains(cand_top, 5, ref_top[i])) top5_overlap++;
        if (i < 10 && topk_contains(cand_top, 10, ref_top[i])) top10_overlap++;
        if (topk_contains(cand_top, 20, ref_top[i])) top20_overlap++;
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
        const float delta = cand[i] - ref[i];
        const float abs_delta = fabsf(delta);
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
        .top10_overlap = top10_overlap,
        .top20_overlap = top20_overlap,
        .nonfinite = nonfinite,
        .rms = (float)sqrt(sumsq / (double)vocab),
        .max_abs = max_abs,
        .top20_max_abs = top20_max_abs,
    };
    return out;
}

static void build_followup_prompt(ds4_engine *engine,
                                  const ds4_tokens *base_prompt,
                                  const int *assistant_tokens,
                                  int assistant_count,
                                  const char *followup_prompt,
                                  ds4_tokens *turn1_out,
                                  ds4_tokens *out) {
    ds4_tokens_copy(turn1_out, base_prompt);
    for (int i = 0; i < assistant_count; i++) {
        ds4_tokens_push(turn1_out, assistant_tokens[i]);
    }
    ds4_tokens_copy(out, turn1_out);
    ds4_chat_append_message(engine, out, "user", followup_prompt);
    ds4_chat_append_assistant_prefix(engine, out, DS4_THINK_NONE);
}

static void print_tokens(const int *tokens, int n) {
    putchar('[');
    for (int i = 0; i < n; i++) {
        if (i) putchar(',');
        printf("%d", tokens[i]);
    }
    putchar(']');
}

static bool tokens_equal(const int *a, int na, const int *b, int nb) {
    if (na != nb) return false;
    for (int i = 0; i < na; i++) if (a[i] != b[i]) return false;
    return true;
}

int main(int argc, char **argv) {
    cfg_t cfg = {
        .system_prompt = "You are a concise assistant.",
        .followup_prompt = "Continue with one more short sentence that depends on your previous answer.",
        .ctx_size = 16384,
        .gen_tokens = 8,
        .temperature = 0.0f,
        .top_p = 1.0f,
        .min_p = 0.0f,
        .seed = 1u,
    };
    parse_args(&cfg, argc, argv);

    char *prompt_text = cfg.prompt_file ? read_file(cfg.prompt_file)
                                        : strdup("Write one short sentence about distributed systems.");
    if (!prompt_text) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: failed to read prompt\n");
        return 1;
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
    };

    ds4_engine *engine = NULL;
    ds4_session *base = NULL;
    ds4_session *prefill = NULL;
    ds4_session *replay = NULL;
    ds4_tokens prompt1 = {0};
    ds4_tokens prompt1_turn1 = {0};
    ds4_tokens prompt2 = {0};
    token_run turn1 = {0};
    token_run turn2_prefill = {0};
    token_run turn2_replay = {0};
    float *prefill_turn1_logits = NULL;
    float *replay_turn1_logits = NULL;
    float *prefill_seed_logits = NULL;
    float *replay_seed_logits = NULL;
    int rc = 1;
    char err[256] = {0};

    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: failed to open engine\n");
        goto cleanup;
    }
    if (ds4_session_create(&base, engine, cfg.ctx_size) != 0 || !base) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: failed to create base session\n");
        goto cleanup;
    }
    ds4_encode_chat_prompt(engine, cfg.system_prompt, prompt_text, DS4_THINK_NONE, &prompt1);
    if (prompt1.len <= 0) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: prompt tokenization failed\n");
        goto cleanup;
    }
    if (ds4_session_sync(base, &prompt1, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: base sync failed: %s\n",
                err[0] ? err : "unknown error");
        goto cleanup;
    }
    if (!run_tokens(engine, base, &cfg, &turn1)) goto cleanup;

    build_followup_prompt(engine,
                          &prompt1,
                          turn1.tokens,
                          turn1.token_count,
                          cfg.followup_prompt,
                          &prompt1_turn1,
                          &prompt2);

    const int vocab = ds4_engine_vocab_size(engine);
    prefill_turn1_logits = malloc((size_t)vocab * sizeof(prefill_turn1_logits[0]));
    replay_turn1_logits = malloc((size_t)vocab * sizeof(replay_turn1_logits[0]));
    prefill_seed_logits = malloc((size_t)vocab * sizeof(prefill_seed_logits[0]));
    replay_seed_logits = malloc((size_t)vocab * sizeof(replay_seed_logits[0]));
    if (!prefill_turn1_logits || !replay_turn1_logits || !prefill_seed_logits || !replay_seed_logits) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: out of memory\n");
        goto cleanup;
    }

    if (ds4_session_create(&prefill, engine, cfg.ctx_size) != 0 || !prefill ||
        ds4_session_create(&replay, engine, cfg.ctx_size) != 0 || !replay) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: failed to create comparison sessions\n");
        goto cleanup;
    }
    if (ds4_session_sync(prefill, &prompt1_turn1, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: prefill turn1 sync failed: %s\n",
                err[0] ? err : "unknown error");
        goto cleanup;
    }
    if (ds4_session_sync(replay, &prompt1, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: replay prompt1 sync failed: %s\n",
                err[0] ? err : "unknown error");
        goto cleanup;
    }
    for (int i = 0; i < turn1.token_count; i++) {
        if (ds4_session_eval(replay, turn1.tokens[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase55-prefill-vs-decode: replay turn1 eval failed: %s\n",
                    err[0] ? err : "unknown error");
            goto cleanup;
        }
    }
    if (ds4_session_copy_logits(prefill, prefill_turn1_logits, vocab) != vocab ||
        ds4_session_copy_logits(replay, replay_turn1_logits, vocab) != vocab) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: failed to copy turn1 logits\n");
        goto cleanup;
    }
    parity_metrics turn1_cmp = compare_logits(prefill_turn1_logits, replay_turn1_logits, vocab);
    printf("PHASE55_PREFILL_DECODE_TURN1_ARGMAX prefill=%d replay=%d\n",
           ds4_session_argmax(prefill), ds4_session_argmax(replay));
    printf("PHASE55_PREFILL_DECODE_TURN1_LOGITS top5=%d top10=%d top20=%d rms=%.8f max_abs=%.8f top20_max_abs=%.8f nonfinite=%d\n",
           turn1_cmp.top5_overlap,
           turn1_cmp.top10_overlap,
           turn1_cmp.top20_overlap,
           turn1_cmp.rms,
           turn1_cmp.max_abs,
           turn1_cmp.top20_max_abs,
           turn1_cmp.nonfinite);

    if (ds4_session_sync(prefill, &prompt2, err, sizeof(err)) != 0 ||
        ds4_session_sync(replay, &prompt2, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: prompt2 sync failed: %s\n",
                err[0] ? err : "unknown error");
        goto cleanup;
    }
    if (ds4_session_copy_logits(prefill, prefill_seed_logits, vocab) != vocab ||
        ds4_session_copy_logits(replay, replay_seed_logits, vocab) != vocab) {
        fprintf(stderr, "issue304-phase55-prefill-vs-decode: failed to copy seed logits\n");
        goto cleanup;
    }
    parity_metrics seed_cmp = compare_logits(prefill_seed_logits, replay_seed_logits, vocab);
    printf("PHASE55_PREFILL_DECODE_ARGMAX prefill=%d replay=%d\n",
           ds4_session_argmax(prefill), ds4_session_argmax(replay));
    printf("PHASE55_PREFILL_DECODE_LOGITS top5=%d top10=%d top20=%d rms=%.8f max_abs=%.8f top20_max_abs=%.8f nonfinite=%d\n",
           seed_cmp.top5_overlap,
           seed_cmp.top10_overlap,
           seed_cmp.top20_overlap,
           seed_cmp.rms,
           seed_cmp.max_abs,
           seed_cmp.top20_max_abs,
           seed_cmp.nonfinite);

    if (!run_tokens(engine, prefill, &cfg, &turn2_prefill) ||
        !run_tokens(engine, replay, &cfg, &turn2_replay)) {
        goto cleanup;
    }
    printf("PHASE55_PREFILL_DECODE_TURN2 prefill=");
    print_tokens(turn2_prefill.tokens, turn2_prefill.token_count);
    printf(" replay=");
    print_tokens(turn2_replay.tokens, turn2_replay.token_count);
    printf("\n");

    printf("PHASE55_PREFILL_DECODE_RESULT %s turn1_tokens=%d turn2_tokens=%d\n",
           tokens_equal(turn2_prefill.tokens,
                        turn2_prefill.token_count,
                        turn2_replay.tokens,
                        turn2_replay.token_count) ? "pass" : "mismatch",
           turn1.token_count,
           turn2_prefill.token_count);
    rc = 0;

cleanup:
    free(replay_seed_logits);
    free(prefill_seed_logits);
    free(replay_turn1_logits);
    free(prefill_turn1_logits);
    token_run_free(&turn2_replay);
    token_run_free(&turn2_prefill);
    token_run_free(&turn1);
    ds4_tokens_free(&prompt2);
    ds4_tokens_free(&prompt1_turn1);
    ds4_tokens_free(&prompt1);
    ds4_session_free(replay);
    ds4_session_free(prefill);
    ds4_session_free(base);
    ds4_engine_close(engine);
    free(prompt_text);
    return rc;
}
