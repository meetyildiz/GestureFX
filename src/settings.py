drumrack_grid_size = (3, 3)
pianoroll_grid_size=(6, 2)
session_grid_size=(6, 6)
mixing_grid_size=(6, 1)
effects_grid_size=(2, 2)

screen_modes_list = ["DRUMRACK", "PIANOROLL", "MIXING", "EFFECTS"]

screen_modes_status = {
        "DRUMRACK": {
            "PAGE":3
        },
        "PIANOROLL": {
            "PAGE":3
        },
        "SESSION": {
            "PAGE":1
        },
        "MIXING": {
            "PAGE":1
        },
        "EFFECTS": {
            "PAGE":1
        },
    }


last_action_index = {"RIGHT": None, "LEFT": None}
last_hands_gestures = {"RIGHT": None, "LEFT": None}
last_channel_index = {"RIGHT": None, "LEFT": None}
last_screen_mode = "EFFECTS"
current_screen_mode = "EFFECTS"



#mixing_grid_size[0]


effects_cache = [(0,0) for i in range(127)]