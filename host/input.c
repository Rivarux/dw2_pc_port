#include <stdio.h>
#include <string.h>

#include "host/host.h"
#include "host/host_sdl.h"
#include "host/settings.h"
#include "host/ui.h"

/* Host input state: keyboard and SDL gamepads as PS1 digital pad button masks, per port. Read
 * after each event pump; changes are logged. P1.7 turns them into pad replies in Pad_RecvBufs.
 * PG.10 b3: the window side (events, device state: Host_InputDevice / Press / Poll) runs on the
 * main thread, the game's view (Host_InputLatch, Host_PadButtons, scripts) on the game thread;
 * held and tapped buttons pass between them under a lock, latched once per VBlank. */

#define PORTS 2
#define TRIGGER_ON 16384 /* trigger axis 0..32767: L2 / R2 held past half */

#define STICK_ON 16384   /* left stick past half: D-pad (stick_dpad setting) */

#define SELECT_BIT 0          /* PAD_SELECT: its gamepad binding is Settings_Pad(SELECT_BIT) */
#define SELECT_HOLD_MS 1000  /* PR.38: Select held this long opens the F1 window */
#define SELECT_TAP_MS 80     /* PR.38: a short Select press reaches the game for this long */

/* PR.2: the key and gamepad bindings come from the settings (host/settings.c; defaults = the
 * tables this file had before: arrows, Z X A S, Q W E R, Return, Backspace / Right Shift; the
 * gamepad by position, south = Cross, triggers = L2 / R2). Esc quits (host/sdl.c). While the F1
 * settings window is open the game sees no keyboard or gamepad (scripted input still runs). */

static const char *const bit_names[16] = {
    "Select", "L3", "R3", "Start", "Up", "Right", "Down", "Left",
    "L2", "R2", "L1", "R1", "Triangle", "Circle", "Cross", "Square",
};

static SDL_Gamepad *pads[PORTS];
static SDL_AtomicInt pad_on[PORTS]; /* pads[p] != NULL, for the game thread */
static unsigned short buttons[PORTS]; /* the game's view, latched per VBlank */
static SDL_Mutex *lock;
static unsigned short held[PORTS];    /* keyboard + gamepads at the last poll (under lock) */
static unsigned short pressed[PORTS]; /* pressed since the last latch: a tap shorter than one
                                       * VBlank still shows for one latch (under lock) */

/* --press / --press2: scripted button holds on port 0 / 1 by VBlank wait. */
#define SCRIPT_MAX 4096
static struct {
    unsigned int at, len;
    unsigned short bits;
    int port;
} script[SCRIPT_MAX];
static int script_count;
static unsigned short script_bits[PORTS];
static int script_port1; /* a --press2 entry exists: port 1 is a scripted pad */
static int pad2_keys;    /* --pad2-keys: port 1 connected, Tab moves the keyboard between ports */
static int kbd_port;     /* port the keyboard drives */

/* PR.38 (issue #5): with the select_hold_menu setting, Select (a gamepad's Back / Select button,
 * or a key bound to Select: Backspace, Right Shift) opens the F1 window when held for
 * SELECT_HOLD_MS. The game sees that Select only when it is let go before then, as a short press
 * (SELECT_TAP_MS) at the release. Select, not Start: the game reads a held Start (dungeon automap
 * colours, src/stag4000/automap.c), Select only as a press. Per gamepad port, plus KBD for the
 * keyboard (its tap goes to kbd_port). Main thread. */
#define KBD PORTS
static struct {
    int down;   /* Select is down since t */
    int opened; /* this hold opened the F1 window */
    Uint64 t, tap_end;
} select_hold[PORTS + 1];

/* The gamepad button bound to Select while the hold is on, else -1 (off, or bound to a trigger). */
static int select_hold_button(void) {
    int c = Settings_Pad(SELECT_BIT);

    return Settings_Get(SET_SELECT_HOLD) && c >= 0 && c < SDL_GAMEPAD_BUTTON_COUNT ? c : -1;
}

int Host_InputSelectHoldOn(void) {
    return select_hold_button() >= 0 && (pads[0] != NULL || pads[1] != NULL);
}

static void select_down(int i) {
    select_hold[i].down = 1;
    select_hold[i].opened = 0;
    select_hold[i].t = SDL_GetTicks();
}

static void select_up(int i) {
    if (select_hold[i].down) {
        select_hold[i].down = 0;
        if (!select_hold[i].opened && !Ui_BlocksGameInput()) {
            select_hold[i].tap_end = SDL_GetTicks() + SELECT_TAP_MS;
        }
    }
}

unsigned short Host_PadButtons(int port) {
    return port >= 0 && port < PORTS ? buttons[port] : 0;
}

int Host_PadConnected(int port) {
    return port == 0 || (port == 1 && (SDL_GetAtomicInt(&pad_on[1]) || script_port1 || pad2_keys));
}

void Host_InputPad2Keys(void) {
    pad2_keys = 1;
}

int Host_InputScriptAdd(int port, unsigned int at, unsigned short bits, unsigned int len) {
    if (script_count == SCRIPT_MAX || port < 0 || port >= PORTS) {
        return 0;
    }
    if (port == 1) {
        script_port1 = 1;
    }
    script[script_count].port = port;
    script[script_count].at = at;
    script[script_count].len = len;
    script[script_count].bits = bits;
    script_count++;
    return 1;
}

void Host_InputScriptTick(unsigned int wait) {
    int i;

    script_bits[0] = script_bits[1] = 0;
    for (i = 0; i < script_count; i++) {
        if (wait >= script[i].at && wait < script[i].at + script[i].len) {
            script_bits[script[i].port] |= script[i].bits;
        }
    }
}

unsigned short Host_PadParseButtons(const char *names) {
    unsigned short m = 0;
    const char *s = names;

    while (*s != 0) {
        size_t n = strcspn(s, "+");
        int i;

        for (i = 0; i < 16; i++) {
            if (strlen(bit_names[i]) == n && SDL_strncasecmp(s, bit_names[i], n) == 0) {
                break;
            }
        }
        if (i == 16) {
            return 0;
        }
        m |= (unsigned short)(1 << i);
        s += n;
        if (*s == '+') {
            s++;
        }
    }
    return m;
}

void Host_InputDevice(const SDL_Event *e) {
    SDL_JoystickID id = e->gdevice.which;
    int p;

    if (e->type == SDL_EVENT_GAMEPAD_ADDED) {
        for (p = 0; p < PORTS; p++) {
            if (pads[p] != NULL && SDL_GetGamepadID(pads[p]) == id) {
                return;
            }
        }
        for (p = 0; p < PORTS && pads[p] != NULL; p++) {
        }
        if (p == PORTS) {
            printf("[input] gamepad %u ignored (both ports taken)\n", (unsigned)id);
            return;
        }
        pads[p] = SDL_OpenGamepad(id);
        if (pads[p] == NULL) {
            printf("[input] gamepad %u open failed: %s\n", (unsigned)id, SDL_GetError());
            return;
        }
        SDL_SetAtomicInt(&pad_on[p], 1);
        printf("[input] gamepad %u on port %d: %s\n", (unsigned)id, p, SDL_GetGamepadName(pads[p]));
    } else {
        for (p = 0; p < PORTS; p++) {
            if (pads[p] != NULL && SDL_GetGamepadID(pads[p]) == id) {
                SDL_CloseGamepad(pads[p]);
                pads[p] = NULL;
                SDL_SetAtomicInt(&pad_on[p], 0);
                SDL_zero(select_hold[p]);
                printf("[input] gamepad %u removed from port %d\n", (unsigned)id, p);
            }
        }
    }
    fflush(stdout);
}

/* The pad bits bound to scancode `sc` (keyboard) or gamepad button `b`. */
static unsigned short key_bits(int sc) {
    unsigned short m = 0;
    int bit, slot;

    for (bit = 0; bit < 16; bit++) {
        for (slot = 0; slot < SET_KEY_SLOTS; slot++) {
            if (Settings_Key(bit, slot) == sc) {
                m |= (unsigned short)(1 << bit);
            }
        }
    }
    return m;
}

static unsigned short pad_bits(int b) {
    unsigned short m = 0;
    int bit;

    for (bit = 0; bit < 16; bit++) {
        if (Settings_Pad(bit) == b) {
            m |= (unsigned short)(1 << bit);
        }
    }
    return m;
}

static void lock_init(void) {
    if (lock == NULL) {
        lock = SDL_CreateMutex();
    }
}

void Host_InputInit(void) {
    lock_init();
}

void Host_InputPress(const SDL_Event *e) {
    unsigned short bits[PORTS] = { 0, 0 };
    int p;

    if (e->type == SDL_EVENT_KEY_DOWN) {
        if (e->key.scancode == SDL_SCANCODE_TAB && pad2_keys && !e->key.repeat) {
            kbd_port ^= 1;
            printf("[input] keyboard on port %d\n", kbd_port);
            fflush(stdout);
            return;
        }
        bits[kbd_port] = key_bits(e->key.scancode);
        if ((bits[kbd_port] & PAD_SELECT) && Settings_Get(SET_SELECT_HOLD)) {
            if (!e->key.repeat && !Ui_BlocksGameInput()) {
                select_down(KBD);
            }
            bits[kbd_port] &= (unsigned short)~PAD_SELECT; /* sent at the release, if short */
        }
    } else {
        for (p = 0; p < PORTS; p++) {
            if (pads[p] != NULL && SDL_GetGamepadID(pads[p]) == e->gbutton.which) {
                bits[p] = pad_bits(e->gbutton.button);
                if (e->gbutton.button == select_hold_button() && !Ui_BlocksGameInput()) {
                    select_down(p);
                    bits[p] &= (unsigned short)~PAD_SELECT; /* sent at the release, if short */
                }
            }
        }
    }
    if (Ui_BlocksGameInput()) {
        return;
    }
    lock_init();
    SDL_LockMutex(lock);
    for (p = 0; p < PORTS; p++) {
        pressed[p] |= bits[p];
    }
    SDL_UnlockMutex(lock);
}

void Host_InputRelease(const SDL_Event *e) {
    int p;

    if (e->type == SDL_EVENT_KEY_UP) {
        if (key_bits(e->key.scancode) & PAD_SELECT) {
            select_up(KBD);
        }
        return;
    }
    for (p = 0; p < PORTS; p++) {
        if (pads[p] != NULL && SDL_GetGamepadID(pads[p]) == e->gbutton.which &&
            e->gbutton.button == select_hold_button()) {
            select_up(p);
        }
    }
}

int Host_InputSelectKeyHeld(const SDL_Event *e) {
    if ((e->type != SDL_EVENT_KEY_DOWN && e->type != SDL_EVENT_KEY_UP) || !select_hold[KBD].down ||
        !select_hold[KBD].opened || !(key_bits(e->key.scancode) & PAD_SELECT)) {
        return 0;
    }
    if (e->type == SDL_EVENT_KEY_UP) {
        select_hold[KBD].down = 0;
    }
    return 1;
}

static void log_buttons(int port, unsigned short b) {
    int i;

    printf("[input] port %d: %04X", port, b);
    for (i = 0; i < 16; i++) {
        if (b & (1 << i)) {
            printf(" %s", bit_names[i]);
        }
    }
    printf("\n");
    fflush(stdout);
}

void Host_InputPoll(void) {
    const bool *keys = SDL_GetKeyboardState(NULL);
    unsigned short b[PORTS] = { 0, 0 };
    int bit, slot, p;
    int hold_button = select_hold_button();
    Uint64 now = SDL_GetTicks();

    for (p = 0; p <= KBD; p++) {
        if (select_hold[p].down && !select_hold[p].opened && now - select_hold[p].t >= SELECT_HOLD_MS) {
            select_hold[p].opened = 1;
            if (!Ui_Headless() && !SettingsUi_IsOpen()) {
                SettingsUi_SetOpen(1);
                if (p == KBD) {
                    printf("[input] Select held on the keyboard: settings\n");
                } else {
                    printf("[input] Select held on port %d: settings\n", p);
                }
                fflush(stdout);
            }
        }
    }
    if (Ui_BlocksGameInput()) {
        goto store; /* F1 open, or an ImGui text field types (dev overlay) */
    }
    for (bit = 0; bit < 16; bit++) {
        for (slot = 0; slot < SET_KEY_SLOTS; slot++) {
            int sc = Settings_Key(bit, slot);

            if (sc > 0 && sc < SDL_SCANCODE_COUNT && keys[sc]) {
                b[kbd_port] |= (unsigned short)(1 << bit);
            }
        }
    }
    if (Settings_Get(SET_SELECT_HOLD)) {
        /* PR.38: a held Select key stays away from the game; a short press shows after the release */
        b[kbd_port] &= (unsigned short)~PAD_SELECT;
        if (now < select_hold[KBD].tap_end) {
            b[kbd_port] |= PAD_SELECT;
        }
    }
    for (p = 0; p < PORTS; p++) {
        if (pads[p] == NULL) {
            continue;
        }
        for (bit = 0; bit < 16; bit++) {
            int c = Settings_Pad(bit);

            if ((c >= 0 && c < SDL_GAMEPAD_BUTTON_COUNT && SDL_GetGamepadButton(pads[p], (SDL_GamepadButton)c)) ||
                (c == SET_PAD_LTRIGGER && SDL_GetGamepadAxis(pads[p], SDL_GAMEPAD_AXIS_LEFT_TRIGGER) >= TRIGGER_ON) ||
                (c == SET_PAD_RTRIGGER && SDL_GetGamepadAxis(pads[p], SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) >= TRIGGER_ON)) {
                b[p] |= (unsigned short)(1 << bit);
            }
        }
        if (Settings_Get(SET_STICK_DPAD)) {
            int x = SDL_GetGamepadAxis(pads[p], SDL_GAMEPAD_AXIS_LEFTX);
            int y = SDL_GetGamepadAxis(pads[p], SDL_GAMEPAD_AXIS_LEFTY);

            b[p] |= (x <= -STICK_ON ? PAD_LEFT : 0) | (x >= STICK_ON ? PAD_RIGHT : 0) | (y <= -STICK_ON ? PAD_UP : 0) |
                    (y >= STICK_ON ? PAD_DOWN : 0);
        }
        if (hold_button >= 0) {
            /* PR.38: a held Select stays away from the game; a short press shows after the release */
            if (SDL_GetGamepadButton(pads[p], (SDL_GamepadButton)hold_button)) {
                b[p] &= (unsigned short)~PAD_SELECT;
            }
            if (now < select_hold[p].tap_end) {
                b[p] |= PAD_SELECT;
            }
        }
    }
store:
    lock_init();
    SDL_LockMutex(lock);
    for (p = 0; p < PORTS; p++) {
        held[p] = b[p];
    }
    SDL_UnlockMutex(lock);
}

void Host_InputLatch(void) {
    unsigned short b[PORTS];
    int p;

    lock_init();
    SDL_LockMutex(lock);
    for (p = 0; p < PORTS; p++) {
        b[p] = held[p] | pressed[p];
        pressed[p] = 0;
    }
    SDL_UnlockMutex(lock);
    for (p = 0; p < PORTS; p++) {
        b[p] |= script_bits[p];
        if (b[p] != buttons[p]) {
            buttons[p] = b[p];
            log_buttons(p, b[p]);
        }
    }
}

void Host_InputClose(void) {
    int p;

    for (p = 0; p < PORTS; p++) {
        if (pads[p] != NULL) {
            SDL_CloseGamepad(pads[p]);
            pads[p] = NULL;
        }
    }
}
