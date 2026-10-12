#ifndef BACKEND_PSXGPU_HD_H
#define BACKEND_PSXGPU_HD_H

#include <stdint.h>

/* HD output of the emulated GPU (PG.1): the draw / display buffers drawn a second time at
 * S x S pixels per VRAM pixel ("surfaces"), next to the 1x VRAM, which stays exactly as the PS1
 * GPU would leave it (textures, CLUTs, CPU access and the classic picture). Textures are always
 * read from the 1x VRAM: the game never samples a draw buffer (PG.0 finding). With scale 1 the
 * HD path is off and nothing here runs. backend/psxgpu.c calls these after each 1x operation. */

/* Drawing state of backend/psxgpu.c (E1-E6). */
typedef struct {
    int clip_x0, clip_y0, clip_x1, clip_y1; /* draw area, inclusive */
    int ofs_x, ofs_y;
    uint32_t tpage;                         /* E1 bits 0-8 and 11 */
    int dither, draw_disp, flip_x, flip_y;
    int tw_mask_x, tw_mask_y, tw_off_x, tw_off_y;
    int set_mask, check_mask;
} PsxGpuState;

/* One polygon vertex in VRAM coordinates (drawing offset applied), colour 8 bits per channel;
 * px, py, pz: the precise position and depth (PG.2) when the polygon is drawn precise. PR.34
 * smooth frames: src = its PGXP source (-1 none), qx, qy, qz = the precise position of the same
 * vertex in the frame before (filled when the frame ends, PsxHdCmd.blend). */
typedef struct {
    int x, y;
    int r, g, b;
    int u, v;
    float px, py, pz;
    int src;
    float qx, qy, qz;
} PsxVtx;

/* Texture of a primitive: mode 0 4-bit, 1 8-bit, 2 15-bit; page tx / ty, CLUT cx / cy. */
typedef struct {
    int mode;
    int tx, ty, cx, cy;
    int raw;
} PsxTex;

/* Scale 1..8. The HD path is on with a scale above 1 or 16:9. Surfaces are rebuilt from the 1x
 * VRAM at the next use. */
void PsxHd_SetScale(int scale);
int PsxHd_Scale(void);
int PsxHd_On(void);
/* Drawing threads for the HD surfaces (0 = one per logical core, at most 8). Takes effect when
 * the first HD frame is drawn. */
void PsxHd_SetThreads(int n);
/* PG.3 16:9: wide surfaces (margins of PsxHd_Margin(w) VRAM pixels per side, 0 when off).
 * PsxHd_PillarboxFrame: the frame being built keeps black sides (2D-only pictures);
 * PsxHd_BeginFrame (libgpu DrawOTag) starts drawing a frame and takes that mark. */
void PsxHd_SetWide(int on);
int PsxHd_Wide(void);
int PsxHd_Margin(int w);
void PsxHd_PillarboxFrame(void);
void PsxHd_BeginFrame(void);
/* PG.2 no-wobble geometry: polygons whose vertices all have a precise GTE position
 * (backend/pgxp.c) are drawn there, with perspective-correct texture mapping. */
void PsxHd_SetPgxp(int on);
int PsxHd_Pgxp(void);

/* PR.2b sharpening of the HD picture (contrast adaptive, strength 0..100, 0 = off): applied by
 * PsxHd_ReadDisplay and by the host's GPU publish (PsxHw_Sharpen), only with a scale above 1
 * (PsxHd_SharpenActive is 0 at scale 1: the 1x picture is never sharpened). A change bumps
 * PsxHd_Serial. Game thread. */
void PsxHd_SetSharpen(int strength);
int PsxHd_Sharpen(void);
int PsxHd_SharpenActive(void);
/* fn(part, parts, arg) for every part on the HD drawing threads and the calling thread (after a
 * flush); returns when all parts are done. */
void PsxHd_Parallel(void (*fn)(int part, int parts, void *arg), void *arg);
/* backend/sharpen.c: in (w x h, 0xAARRGGBB, alpha kept) sharpened into out, rows in parallel. */
void PsxSharpen_Cpu(const uint32_t *in, uint32_t *out, int w, int h, int strength);

/* PG.10b GPU renderer (backend/psxgpu_hw.c, needs PsxHw_Init first): surfaces drawn on the GPU
 * instead of the software rasterizer; surfaces are rebuilt from the 1x VRAM on a switch. */
void PsxHd_SetGpu(int on);
/* backend/psxgpu_hw.c: GPU renderer on an SDL_GPUDevice made for SPIR-V (1 = ready); shutdown
 * after PsxHd_SetGpu(0), before the device goes. */
int PsxHw_Init(void *device);
void PsxHw_Shutdown(void);
/* PG.10 b3: pictures for the host's present (another thread): a GPU texture w x h (render target
 * + sampler), its release (deferred by SDL_GPU until the GPU is done with it), and a copy of the
 * texture rect x, y, w, h of src to the top-left corner of dst, submitted on the calling thread. */
void *PsxHw_Create(int w, int h);
void PsxHw_Release(void *tex);
void PsxHw_Copy(void *src, int x, int y, int w, int h, void *dst);
/* PR.2b: as PsxHw_Copy, sharpened (strength 1..100, backend/shaders/sharpen.frag). */
void PsxHw_Sharpen(void *src, int x, int y, int w, int h, void *dst, int strength);
int PsxHd_Gpu(void);
/* Draws everything queued (before the 1x VRAM changes by a transfer: queued primitives see the
 * textures as they were when the packet was sent). */
void PsxHd_Sync(void);

/* A draw or display buffer rect (PutDrawEnv clip, PutDispEnv area). Kept when an existing
 * surface holds it; otherwise the surfaces it overlaps are dropped and it becomes one. */
void PsxHd_Register(int x, int y, int w, int h);

/* Primitives, drawn into the surface that holds the current draw area. precise: use px / py /
 * pz (decided per polygon by the caller, so both triangles of a quad agree). */
void PsxHd_Triangle(const PsxGpuState *st, const PsxVtx *v0, const PsxVtx *v1, const PsxVtx *v2, int gouraud,
                    const PsxTex *t, int semi, int abr, int precise);
void PsxHd_Rect(const PsxGpuState *st, int x0, int y0, int w, int h, const PsxTex *t, int u0, int v0, int r, int g,
                int b, int semi, int abr);
void PsxHd_Line(const PsxGpuState *st, const PsxVtx *a, const PsxVtx *b, int gouraud, int semi, int abr);

/* Transfers: a fill (colour in VRAM format), or a rect the 1x VRAM just got from the CPU or a
 * VRAM copy (copied into the surfaces, nearest). */
void PsxHd_Fill(int x, int y, int w, int h, uint16_t c);
void PsxHd_Refresh(int x, int y, int w, int h);

/* The display area (x, y, w, h in VRAM) from a surface: w*S x h*S pixels of 0x00RRGGBB into out
 * (NULL: size only), (w + 2 M)*S wide with the 16:9 margins when it is a whole buffer. Returns 0
 * when HD is off or no surface holds the area. */
int PsxHd_ReadDisplay(int x, int y, int w, int h, uint32_t *out, int *ow, int *oh);
/* GPU renderer: the display area's surface texture (an SDL_GPUTexture, tw x th) and the source
 * rect of the picture in it (with the margins when it is a whole buffer), everything queued
 * submitted. Returns 0 when the GPU renderer is off or no surface holds the area. */
int PsxHd_DisplayTexture(int x, int y, int w, int h, void **tex, int *tw, int *th, int *sx, int *sy, int *sw,
                         int *sh);
/* Margin on each side of the last PsxHd_ReadDisplay picture, in its pixels (0: plain 4:3). */
int PsxHd_LastMargin(void);
/* Changes on every surface write or display change (the host skips unchanged frames). */
unsigned PsxHd_Serial(void);

/* PR.34 smooth frames (GPU renderer with PGXP; the city first). With smooth on, each frame
 * (DrawOTag) is logged; PsxHd_EndFrame (libgpu, after Pgxp_EndFrame) turns a frame that can be
 * drawn again into a package: its commands with the frame-before position of every 3D vertex
 * (Pgxp_PrevPos) and the scroll of its 2D layers, a copy of the VRAM, its picture as the base and
 * the time of its VBlank. Only frames with a 2D layer (PsxHd_SmoothLayer: the city map) or a scene
 * hint (PsxHd_SmoothScene: the dungeon) are packaged; the others are shown as they are. Fills
 * queued just before DrawOTag (PutDrawEnv's background clear) start the frame's log. Game thread: PsxHd_SetSmooth,
 * PsxHd_SmoothLayer, PsxHd_SetFrameTime (the deadline of the VBlank being run), PsxHd_EndFrame.
 * PsxHd_Packet: the address of the packet libgpu's DrawOTag sends (0 outside it).
 * Window thread: PsxHd_SmoothPicture draws the newest package at time t (ns, SDL_GetTicksNS):
 * blended from the frame before (alpha 0) to it (alpha 1) by t between its VBlank and the next
 * one, sharpened like a published picture; tex (an SDL_GPUTexture w x h, margin as
 * PsxHd_LastMargin) stays valid until the next call. 0: no package, show the published picture.
 * PsxHd_SmoothAlpha: the alpha of the last PsxHd_SmoothPicture (1 when it returned 0).
 * PsxHd_SmoothRead (headless checks, game thread): the newest package at alpha, read back as
 * 0x00RRGGBB (out NULL: size only). */
extern uintptr_t PsxHd_Packet;
void PsxHd_SetSmooth(int on);
int PsxHd_Smooth(void);
int PsxHd_SmoothActive(void);
void PsxHd_SmoothLayer(int id, uintptr_t start, uintptr_t end, int x, int y);
/* PR.34: this frame is packaged without a 2D layer (the dungeon). */
void PsxHd_SmoothScene(void);
void PsxHd_SetFrameTime(uint64_t ns);
/* PR.36 (game thread): on = no packages (the speed-up runs: game frames come about as fast as the
 * monitor refreshes, so blending gains nothing and costs the GPU time the game needs). */
void PsxHd_SmoothHold(int on);
void PsxHd_EndFrame(void);
int PsxHd_SmoothPicture(uint64_t t, void **tex, int *w, int *h, int *margin);
float PsxHd_SmoothAlpha(void);
/* The package made by the last PsxHd_EndFrame (game thread) and the one the last
 * PsxHd_SmoothPicture drew (window thread): numbers, 0 none (for host overlays that follow the picture). */
unsigned PsxHd_PackageSeq(void);
unsigned PsxHd_SmoothSeq(void);
int PsxHd_SmoothRead(float alpha, uint32_t *out, int *w, int *h);

#endif /* BACKEND_PSXGPU_HD_H */
