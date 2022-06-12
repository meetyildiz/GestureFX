import imp
from time import sleep
from turtle import pen
import mido

out_port = mido.open_output('GestureFX', virtual=True)
in_port = mido.open_input('GestureFX', virtual=True)


def send_triger_push_midi(last_channel_index, channel_index):

    
    hand_labels = ["RIGHT", "LEFT"]


    for hand_label in hand_labels:
        if (last_channel_index[hand_label] is None and channel_index[hand_label] is not None):
            msg = mido.Message('note_on', channel=0, note=channel_index[hand_label])
            out_port.send(msg)
            print(hand_label + str(msg))
        elif last_channel_index[hand_label] is not None and channel_index[hand_label] is None:
            msg = mido.Message('note_off', channel=0, note=last_channel_index[hand_label])
            out_port.send(msg)
            print(hand_label + str(msg))
        
        elif last_channel_index[hand_label] != channel_index[hand_label]:
            msg = mido.Message('note_off', channel=0, note=last_channel_index[hand_label])
            out_port.send(msg)
            print(hand_label + str(msg))
            msg = mido.Message('note_on', channel=0, note=channel_index[hand_label])
            out_port.send(msg)
            print(hand_label + str(msg))

        else:
            print(hand_label + " No midi send")






def send_sensitive_push_midi(last_channel_index, channel_index, action_index):

    
    hand_labels = ["RIGHT", "LEFT"]


    for hand_label in hand_labels:
        
        if (last_channel_index[hand_label] is None and channel_index[hand_label] is not None):
            msg = mido.Message('note_on', channel=1, note=channel_index[hand_label], velocity=round(127*action_index[hand_label][3]))
            out_port.send(msg)
            print(hand_label + str(msg))
        elif last_channel_index[hand_label] is not None and channel_index[hand_label] is None:
            msg = mido.Message('note_off', channel=1, note=last_channel_index[hand_label])
            out_port.send(msg)
            print(hand_label + str(msg))
        
        elif last_channel_index[hand_label] != channel_index[hand_label]:
            msg = mido.Message('note_off', channel=1, note=last_channel_index[hand_label])
            out_port.send(msg)
            print(hand_label + str(msg))
            msg = mido.Message('note_on', channel=1, note=channel_index[hand_label], velocity=round(127*action_index[hand_label][3]))
            out_port.send(msg)
            print(hand_label + str(msg))

        else:
            print(hand_label + " No midi send")


def send_effect_midi(channel_index, action_index):

    
    hand_labels = ["RIGHT", "LEFT"]


    for hand_label in hand_labels:
        
        if (channel_index[hand_label] is not None):
            print("GO"*100)
            print(action_index[hand_label])
            msg_x = mido.Message('control_change', channel=2, control=channel_index[hand_label], value=round(127*action_index[hand_label][2]))
            msg_y = mido.Message('control_change', channel=2, control=channel_index[hand_label]+50, value=round(127*action_index[hand_label][3]))

            
            out_port.send(msg_x)
            out_port.send(msg_y)

            print(hand_label + str(msg_x))
            print(hand_label + str(msg_y))

        else:
            print(hand_label + " No midi send")


def send_fader_midi(last_channel_index, channel_index, action_index):
    
    hand_labels = ["RIGHT", "LEFT"]
    for hand_label in hand_labels:
        
        if (channel_index[hand_label] is not None):

            value = round(127*action_index[hand_label][3])
            #value = 50
            msg = mido.Message('control_change', channel=3, control=channel_index[hand_label], value=value)
            out_port.send(msg)
            print(hand_label + str(msg))
        else:
            print(hand_label + " No midi send")