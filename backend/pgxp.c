#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend/pgxp.h"
#include "backend/psxgpu.h"
#include "psyq/gte_core.h"

/* Precise vertices (PG.2, PGXP-style). The C GTE hands every RTPS result over as a float
 * position next to its integer SX / SY (Pgxp_Record through Gte_PreciseHook). The game copies the
 * integers into GPU packets by value, after halving them when the draw area is not 640 / 480 wide
 * / tall (model.c, floor.c), so the entries are keyed by the integer vertex as the packet will
 * hold it, and the float is halved the same way. The integer is the float rounded down, so the
 * float is stored 0.5 lower: the precise picture then sits where the integer one does on
 * average (2D parts and integer vertices line up with it). The GPU looks vertices up by their
 * packet value. Entries live from the projection to the end of the next DrawOTag.
 * PG.10: a key often has several different points in one frame (a city or dungeon frame has
 * hundreds: distinct vertices less than one packet pixel apart, at the same or another depth).
 * Each key keeps up to PG_CAND of them, and a polygon picks one per vertex so that its depths
 * agree (Pgxp_LookupPoly). Before, such keys gave no precise value, and the polygons that used
 * them mixed precise and integer corners with affine texturing, so their texture moved by a
 * different amount than their neighbours' while the camera moved.
 *
 * PR.34 smooth frames (Pgxp_SetSources): each RTPS also gets a source, the input vertex V0 and the
 * projection it went through (an instance: RT, TR, H, OFX, OFY). An instance starts whenever one
 * of those registers changed since the last RTPS; its key names where the matrices came from: the
 * address passed to SetRotMatrix / SetTransMatrix and how many times that address was loaded
 * before in the frame (a bone's viewMat in its model, a stack local reused per actor ...). At the
 * end of a frame each instance looks up the instance with the same key in the frame before, so
 * the position of each vertex in the frame before is known (Pgxp_PrevPos: the vertex of the same
 * rank in that instance, projected with that frame's registers). Every RTPS gets a source (also
 * those near the camera that give no precise point), so the ranks stay the same from frame to
 * frame. Sources live from the projection to the end of the DrawOTag after the next one.
 * PR.34: the dungeon floor grid keeps its matrix and moves its input vertices inside a tile, and
 * its vertex count changes with the view (10 or 11 columns / rows). Pgxp_Lattice marks it: its
 * vertices pair by world point, the change of its first vertex wrapped into half a period.
 * PR.34: the battle camera says each frame whether it moved smoothly (wider jump limits) or was
 * set at once (a cut: nothing blends), Pgxp_Camera. */

#define PG_BITS 15
#define PG_SIZE (1 << PG_BITS)
#define PG_PROBE 16
#define PG_CAND 8

#define MAX_INST 32768
#define MAX_SRC (1 << 18)
#define ORD_BITS 12
#define ORD_SIZE (1 << ORD_BITS)

typedef struct {
    float x, y, z;
    int src; /* index in the frame's sources, -1 none */
} Cand;

typedef struct {
    uint32_t key;
    float rt[9], tr[3]; /* RT as 4.12 / 4096, TR */
    float h, ofx, ofy;
    int prev;     /* the matching instance of the frame before, -1 none */
    int first, n; /* its sources */
    float per[3]; /* Pgxp_Lattice periods, all 0 = no lattice */
    int lattice;
} Inst;

typedef struct {
    int16_t v[3];
    uint8_t shx, shy;
    int inst, k; /* the instance and the rank in it */
} Src;

typedef struct {
    const void *p;
    uint32_t gen, count;
} Ord;

typedef struct {
    Inst *inst;
    Src *src;
    int ninst, nsrc;
    int cam; /* PR.34: Pgxp_Camera this frame (CAM_*) */
} Frame;

enum { CAM_NONE, CAM_STILL, CAM_MOVED, CAM_CUT };

typedef struct {
    uint32_t key, gen;
    int n;
    Cand c[PG_CAND];
} Entry;

static Entry tab[PG_SIZE];
static uint32_t gen = 1;

/* PR.34: the frame being recorded, the frame drawn last and the one before it */
static int sources;
static Frame frames[3];
static Frame *f_cur = &frames[0], *f_last = &frames[1], *f_old = &frames[2];
static uint32_t inst_serial;
static int inst_now = -1;
static const void *rot_p, *tr_p;
static uint32_t rot_n, tr_n, sub_n;
static int lat_on;      /* Pgxp_Lattice for the instances of the matrices loaded last */
static float lat_per[3];
static int cut; /* the frame drawn last is a camera cut (its lattice jumped): no frame-before positions */
static int cam_cur;     /* PR.34: the camera hint of the frame being recorded */
static int cam_prev[CAM_VALUES], cam_nprev = -1;
static Ord ords[ORD_SIZE];
static uint32_t ord_gen = 1;

static uint32_t slot(uint32_t key) {
    return (key * 2654435761u) >> (32 - PG_BITS);
}

static uint32_t make_key(int x, int y) {
    return ((uint32_t)(uint16_t)y << 16) | (uint16_t)x;
}

static uint32_t mix(uint32_t h, uint32_t v) {
    h ^= v + 0x9E3779B9u + (h << 6) + (h >> 2);
    return h;
}

/* how many times p was loaded before in this frame */
static uint32_t ordinal(const void *p) {
    uint32_t k = (uint32_t)(((uintptr_t)p * 2654435761u) >> (32 - ORD_BITS)) & (ORD_SIZE - 1);
    int i;

    for (i = 0; i < 64; i++) {
        Ord *o = &ords[(k + i) & (ORD_SIZE - 1)];
        if (o->gen != ord_gen) {
            o->gen = ord_gen;
            o->p = p;
            o->count = 1;
            return 0;
        }
        if (o->p == p) {
            return o->count++;
        }
    }
    return 0xFFFF;
}

void Pgxp_Matrix(int kind, const void *p) {
    if (!sources) {
        return;
    }
    if (kind == 0) {
        rot_p = p;
        rot_n = ordinal(p);
    } else {
        tr_p = p;
        tr_n = ordinal(p);
    }
    sub_n = 0;
    lat_on = 0;
    inst_serial = Gte_MatSerial - 1; /* a new instance at the next RTPS */
}

void Pgxp_Lattice(int px, int py, int pz) {
    if (!sources) {
        return;
    }
    lat_on = 1;
    lat_per[0] = (float)px;
    lat_per[1] = (float)py;
    lat_per[2] = (float)pz;
    inst_serial = Gte_MatSerial - 1;
}

void Pgxp_Camera(const int *v, int n, int hard) {
    int same;

    if (!sources) {
        return;
    }
    n = n < CAM_VALUES ? n : CAM_VALUES;
    same = n == cam_nprev && memcmp(v, cam_prev, (size_t)n * sizeof(int)) == 0;
    cam_cur = same ? CAM_STILL : hard ? CAM_CUT : CAM_MOVED;
    memcpy(cam_prev, v, (size_t)n * sizeof(int));
    cam_nprev = n;
}

float Pgxp_MoveLimit(void) {
    /* a moving camera turns the battle's narrow lens about 80 VRAM pixels a frame (intro spin) */
    return sources && f_last->cam == CAM_MOVED ? 256.0f : 64.0f;
}

static float s16f(uint32_t w, int hi) {
    return (float)(int16_t)(hi ? w >> 16 : w & 0xFFFF);
}

/* the instance for the GTE state now, or -1 */
static int current_inst(void) {
    Inst *in;
    int k;

    if (inst_now >= 0 && inst_serial == Gte_MatSerial) {
        return inst_now;
    }
    inst_serial = Gte_MatSerial;
    if (f_cur->ninst == MAX_INST) {
        return inst_now = -1;
    }
    in = &f_cur->inst[f_cur->ninst];
    for (k = 0; k < 9; k++) {
        in->rt[k] = s16f(Gte_Cfc2(GTE_RT0 + k / 2), k & 1) / 4096.0f;
    }
    in->tr[0] = (float)(int32_t)Gte_Cfc2(GTE_TRX);
    in->tr[1] = (float)(int32_t)Gte_Cfc2(GTE_TRY);
    in->tr[2] = (float)(int32_t)Gte_Cfc2(GTE_TRZ);
    in->h = (float)(uint16_t)Gte_Cfc2(GTE_H);
    in->ofx = (float)((int32_t)Gte_Cfc2(GTE_OFX) / 65536.0);
    in->ofy = (float)((int32_t)Gte_Cfc2(GTE_OFY) / 65536.0);
    in->key = mix(mix(mix(mix(mix(0, (uint32_t)(uintptr_t)rot_p), rot_n), (uint32_t)(uintptr_t)tr_p), tr_n), sub_n++);
    in->prev = -1;
    in->first = f_cur->nsrc;
    in->n = 0;
    in->lattice = lat_on;
    for (k = 0; k < 3; k++) {
        in->per[k] = lat_on ? lat_per[k] : 0.0f;
    }
    return inst_now = f_cur->ninst++;
}

static int add_src(const int16_t *v0, int shx, int shy) {
    int inst = current_inst();
    Src *s;

    if (inst < 0 || f_cur->nsrc == MAX_SRC) {
        return -1;
    }
    s = &f_cur->src[f_cur->nsrc];
    s->v[0] = v0[0];
    s->v[1] = v0[1];
    s->v[2] = v0[2];
    s->shx = (uint8_t)shx;
    s->shy = (uint8_t)shy;
    s->inst = inst;
    s->k = f_cur->inst[inst].n++;
    return f_cur->nsrc++;
}

void Pgxp_Record(int32_t sxy, double x, double y, double z, int flags, const int16_t *v0) {
    int sx = (int16_t)(sxy & 0xFFFF);
    int sy = (int16_t)((uint32_t)sxy >> 16);
    int w, h, shx, shy, i, j, src = -1;
    uint32_t key, k;
    float fx, fy;

    PsxGpu_DrawAreaSize(&w, &h);
    shx = w != 640;
    shy = h != 480;
    if (sources) {
        src = add_src(v0, shx, shy);
    }
    if (flags & GTE_PRECISE_NEAR) {
        return; /* only the source counts */
    }
    /* the float must describe the same point as the hardware result; when the GTE saturated
     * the integer one, the float is the true position of that packet vertex (PG.10) */
    if (!(flags & GTE_PRECISE_CLAMPED) && !(x > sx - 2.0 && x < sx + 2.0 && y > sy - 2.0 && y < sy + 2.0)) {
        return;
    }
    fx = (float)((shx ? x / 2.0 : x) - 0.5);
    fy = (float)((shy ? y / 2.0 : y) - 0.5);
    key = make_key(sx >> shx, sy >> shy);
    k = slot(key);
    for (i = 0; i < PG_PROBE; i++) {
        Entry *e = &tab[(k + i) & (PG_SIZE - 1)];
        if (e->gen != gen) {
            e->gen = gen;
            e->key = key;
            e->n = 1;
            e->c[0].x = fx;
            e->c[0].y = fy;
            e->c[0].z = (float)z;
            e->c[0].src = src;
            return;
        }
        if (e->key == key) {
            /* the same point again (shared vertex projected twice) */
            for (j = 0; j < e->n; j++) {
                float dx = e->c[j].x - fx, dy = e->c[j].y - fy;
                if (dx * dx + dy * dy <= 1.0f / 4096.0f) {
                    return;
                }
            }
            if (e->n < PG_CAND) {
                e->c[e->n].x = fx;
                e->c[e->n].y = fy;
                e->c[e->n].z = (float)z;
                e->c[e->n].src = src;
                e->n++;
            } else {
                e->n = PG_CAND + 1; /* too many points: no precise value */
            }
            return;
        }
    }
}

static const Entry *find(int x, int y) {
    uint32_t key = make_key(x, y);
    uint32_t k = slot(key);
    int i;

    for (i = 0; i < PG_PROBE; i++) {
        const Entry *e = &tab[(k + i) & (PG_SIZE - 1)];
        if (e->gen != gen) {
            return NULL;
        }
        if (e->key == key) {
            return e->n <= PG_CAND ? e : NULL;
        }
    }
    return NULL;
}

/* relative depth distance */
static float zdist(float a, float b) {
    float d = a > b ? a - b : b - a;
    return d / (a < b ? a : b);
}

/* the candidate of e whose depth is closest to z */
static int nearest(const Entry *e, float z) {
    int j, best = 0;
    float d, bestd = zdist(e->c[0].z, z);

    for (j = 1; j < e->n; j++) {
        d = zdist(e->c[j].z, z);
        if (d < bestd) {
            bestd = d;
            best = j;
        }
    }
    return best;
}

int Pgxp_LookupPoly(int n, const int *x, const int *y, float *fx, float *fy, float *fz, int *src) {
    const Entry *e[4];
    int pick[4];
    int i, j, mask = 0, nref = 0, multi = -1;
    float zref = 0;

    for (i = 0; i < n; i++) {
        e[i] = find(x[i], y[i]);
        pick[i] = 0;
        if (e[i] != NULL) {
            mask |= 1 << i;
            if (e[i]->n == 1) {
                zref += e[i]->c[0].z;
                nref++;
            } else if (multi < 0) {
                multi = i;
            }
        }
    }
    if (multi >= 0 && nref > 0) {
        /* each vertex with several points takes the one nearest the others' mean depth */
        zref /= (float)nref;
        for (i = 0; i < n; i++) {
            if (e[i] != NULL && e[i]->n > 1) {
                pick[i] = nearest(e[i], zref);
            }
        }
    } else if (multi >= 0) {
        /* no single-point corner: try each point of the first such vertex as the depth, the
         * others nearest to it, and keep the set with the smallest depth spread */
        float bestcost = -1;
        int best[4] = { 0, 0, 0, 0 };

        for (j = 0; j < e[multi]->n; j++) {
            float z = e[multi]->c[j].z, zmin = z, zmax = z, cost;

            pick[multi] = j;
            for (i = 0; i < n; i++) {
                if (e[i] != NULL && i != multi) {
                    float zi;

                    pick[i] = nearest(e[i], z);
                    zi = e[i]->c[pick[i]].z;
                    zmin = zi < zmin ? zi : zmin;
                    zmax = zi > zmax ? zi : zmax;
                }
            }
            cost = zdist(zmin, zmax);
            if (bestcost < 0 || cost < bestcost) {
                bestcost = cost;
                for (i = 0; i < n; i++) {
                    best[i] = pick[i];
                }
            }
        }
        for (i = 0; i < n; i++) {
            pick[i] = best[i];
        }
    }
    for (i = 0; i < n; i++) {
        src[i] = -1;
        if (e[i] != NULL) {
            fx[i] = e[i]->c[pick[i]].x;
            fy[i] = e[i]->c[pick[i]].y;
            fz[i] = e[i]->c[pick[i]].z;
            src[i] = e[i]->c[pick[i]].src;
        }
    }
    return mask;
}

void Pgxp_SetSources(int on) {
    int i;

    on = on != 0;
    if (on && frames[0].inst == NULL) {
        for (i = 0; i < 3; i++) {
            frames[i].inst = malloc(MAX_INST * sizeof(Inst));
            frames[i].src = malloc(MAX_SRC * sizeof(Src));
            if (frames[i].inst == NULL || frames[i].src == NULL) {
                printf("[gpu] smooth frames: no memory for the vertex sources\n");
                on = 0;
                break;
            }
        }
    }
    sources = on;
    for (i = 0; i < 3; i++) {
        frames[i].ninst = 0;
        frames[i].nsrc = 0;
    }
    inst_now = -1;
}

/* An instance that moved this far since the frame before is a cut (camera switch, a reused
 * heap slot, a model turned around at once): no blend. A limb of a running model turns up to
 * about 0.5 per frame (PR.34 city walk), so the rotation limit is 1.0. */
static int jumped(const Inst *a, const Inst *b, float tr_limit) {
    int k;

    for (k = 0; k < 9; k++) {
        if (fabsf(a->rt[k] - b->rt[k]) > 1.0f) {
            return 1;
        }
    }
    for (k = 0; k < 3; k++) {
        if (fabsf(a->tr[k] - b->tr[k]) > tr_limit) {
            return 1;
        }
    }
    return a->h != b->h;
}

/* the last frame's instances find theirs in the frame before (by key) */
static void match_last(void) {
    static int head[MAX_INST * 2];
    static int next[MAX_INST];
    const int hsize = MAX_INST * 2;
    Inst *f = f_last->inst, *p = f_old->inst;
    int nf = f_last->ninst, np = f_old->ninst;
    /* PR.34: a moving camera (battle intro spin, eases) moves each instance's translation by
     * more than 2048 a frame (about 0x4E20 away, 6 degrees); a hard-set shot that changed, or the
     * first frame with a camera, is a cut */
    float tr_limit = f_last->cam == CAM_MOVED ? 8192.0f : 2048.0f;
    int i;

    cut = f_last->cam == CAM_CUT || (f_last->cam != CAM_NONE && f_old->cam == CAM_NONE);
    for (i = 0; i < hsize; i++) {
        head[i] = -1;
    }
    for (i = 0; i < np; i++) {
        uint32_t h = p[i].key % (uint32_t)hsize;
        next[i] = head[h];
        head[h] = i;
    }
    for (i = 0; i < nf; i++) {
        int j;

        f[i].prev = -1;
        for (j = head[f[i].key % (uint32_t)hsize]; j >= 0; j = next[j]) {
            if (p[j].key == f[i].key) {
                /* the same vertex count: the ranks pair vertex with vertex (a lattice pairs by
                 * world point, any count) */
                if ((p[j].n == f[i].n || (p[j].lattice && f[i].lattice && p[j].n > 0)) && !jumped(&p[j], &f[i], tr_limit)) {
                    f[i].prev = j;
                } else if (f[i].lattice && p[j].lattice && jumped(&p[j], &f[i], tr_limit)) {
                    /* the lattice holds the camera: it jumped, so the whole frame is a cut */
                    cut = 1;
                }
                break;
            }
        }
    }
}

void Pgxp_EndFrame(void) {
    if (sources) {
        Frame *t = f_old;

        f_cur->cam = cam_cur;
        cam_cur = CAM_NONE;
        f_old = f_last;
        f_last = f_cur;
        f_cur = t;
        f_cur->ninst = 0;
        f_cur->nsrc = 0;
        match_last();
        inst_now = -1;
        rot_p = tr_p = NULL;
        rot_n = tr_n = sub_n = 0;
        lat_on = 0;
        if (++ord_gen == 0) {
            memset(ords, 0, sizeof(ords));
            ord_gen = 1;
        }
    }
    if (++gen == 0) {
        memset(tab, 0, sizeof(tab));
        gen = 1;
    }
}

void Pgxp_CutSrc(int src) {
    if (sources && src >= 0 && src < f_last->nsrc) {
        f_last->inst[f_last->src[src].inst].prev = -1;
    }
}

int Pgxp_PrevPos(int src, float *fx, float *fy, float *fz) {
    const Src *s, *p;
    const Inst *b, *a;
    float m[3], x, y;
    int k;

    if (!sources || cut || src < 0 || src >= f_last->nsrc) {
        return 0;
    }
    s = &f_last->src[src];
    b = &f_last->inst[s->inst];
    if (b->prev < 0) {
        return 0;
    }
    a = &f_old->inst[b->prev];
    if (b->lattice && a->lattice) {
        /* the same world point: this vertex moved by the change of the lattice's first vertex,
         * wrapped into half a period (a step into the next tile renumbers the vertices) */
        const Src *s0 = &f_last->src[b->first], *p0 = &f_old->src[a->first];
        float v[3];

        for (k = 0; k < 3; k++) {
            float d = (float)p0->v[k] - (float)s0->v[k];

            if (b->per[k] > 0) {
                d -= b->per[k] * floorf(d / b->per[k] + 0.5f);
            }
            v[k] = (float)s->v[k] + d;
        }
        for (k = 0; k < 3; k++) {
            m[k] = a->tr[k] + a->rt[k * 3] * v[0] + a->rt[k * 3 + 1] * v[1] + a->rt[k * 3 + 2] * v[2];
        }
    } else {
        p = &f_old->src[a->first + s->k];
        for (k = 0; k < 3; k++) {
            m[k] = a->tr[k] + a->rt[k * 3] * p->v[0] + a->rt[k * 3 + 1] * p->v[1] + a->rt[k * 3 + 2] * p->v[2];
        }
    }
    if (2.0f * m[2] <= a->h) {
        return 0;
    }
    x = m[0] * a->h / m[2] + a->ofx;
    y = m[1] * a->h / m[2] + a->ofy;
    *fx = (s->shx ? x / 2.0f : x) - 0.5f;
    *fy = (s->shy ? y / 2.0f : y) - 0.5f;
    *fz = m[2];
    return 1;
}
