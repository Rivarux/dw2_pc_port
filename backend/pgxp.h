#ifndef BACKEND_PGXP_H
#define BACKEND_PGXP_H

#include <stdint.h>

/* Precise vertices for the HD output (PG.2, backend/pgxp.c). */

/* An RTPS result: the integer SXY2 word and the float position of the same point (GTE screen
 * space, before the game's halving), z = view depth, flags GTE_PRECISE_* (psyq/gte_core.h), v0 =
 * the input vertex. Installed as Gte_PreciseHook. */
void Pgxp_Record(int32_t sxy, double x, double y, double z, int flags, const int16_t *v0);
/* The precise positions for the n (3 or 4) packet vertices of one polygon (drawing offset not
 * applied): bit k of the result is set when vertex k is known (fx / fy / fz [k] filled). A vertex
 * with several points in this frame gets the one whose depth fits the polygon's other corners.
 * src[k]: the point's source for Pgxp_PrevPos (PR.34), -1 none. */
int Pgxp_LookupPoly(int n, const int *x, const int *y, float *fx, float *fy, float *fz, int *src);
/* After each DrawOTag: the projections of the frame just drawn are dropped (their sources stay
 * for Pgxp_PrevPos until the next DrawOTag ends). */
void Pgxp_EndFrame(void);

/* PR.34 smooth frames: keep a source (input vertex + projection) for every RTPS; Pgxp_Matrix is
 * installed as Gte_MatrixHook. After Pgxp_EndFrame, Pgxp_PrevPos gives the position in the frame
 * before of a source of the frame just drawn (as Pgxp_LookupPoly gives positions: halved, 0.5
 * lower, drawing offset not applied): 1, or 0 when it has none (new, cut, other vertex count). */
void Pgxp_SetSources(int on);
void Pgxp_Matrix(int kind, const void *src);
int Pgxp_PrevPos(int src, float *fx, float *fy, float *fz);
/* PR.34: the instance of src is a cut (one of its vertices moved too far): no Pgxp_PrevPos for
 * any of its sources, so a model is blended whole or not at all. */
void Pgxp_CutSrc(int src);
/* PR.34: the projection loaded last is a lattice (the dungeon floor grid): the camera moves its
 * input vertices by less than half a period per frame, px / py / pz (0 = none) per axis. Its
 * vertices pair with the frame before by world point, not by rank (any vertex count). */
void Pgxp_Lattice(int px, int py, int pz);
/* PR.34: the camera of the frame being recorded, its view values v[0..n-1] (n <= CAM_VALUES).
 * hard = a shot set at once (changed values are a cut: no blend in the whole frame); else the
 * camera moves smoothly (eases, spins) and changed values widen the jump limits for that frame.
 * The first frame with a camera after one without is a cut. Pgxp_MoveLimit: the screen move
 * (VRAM pixels) a vertex of the frame drawn last may make and still blend. */
#define CAM_VALUES 16
void Pgxp_Camera(const int *v, int n, int hard);
float Pgxp_MoveLimit(void);

#endif /* BACKEND_PGXP_H */
