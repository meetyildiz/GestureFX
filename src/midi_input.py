from time import sleep
import mido


with mido.open_input() as inport:
    for msg in inport:
        print(msg)




# next
# sysex data=(0,32,107,127,66,2,0,0,25,127) time=0
# sysex data=(0,32,107,127,66,2,0,0,25,0) time=0

# previous
# sysex data=(0,32,107,127,66,2,0,0,24,127) time=0
# sysex data=(0,32,107,127,66,2,0,0,24,0) time=0



# control_change channel=0 control=73 value=69 time=0


#pitch
#pitchwheel channel=0 pitch=-489 time=0
#mod
#control_change channel=0 control=1 value=94 time=0
