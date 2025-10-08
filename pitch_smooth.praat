form Input parameters
    sentence directory ./all_wav/
    sentence outdirectory ./vctk_pitch_out/
    comment plot parameters
    sentence figname mausmooth
    boolean raw 1
    boolean manual 0
    boolean smooth 1
    integer plotmin 75
    integer plotmax 600
    comment advanced extraction parameters
    integer smooth1 15
    integer smooth2 15
endform

Erase all

Create Strings as file list: "pitchlist", outdirectory$+"*.Pitch"
Create Strings as file list: "list", directory$+"*.wav"

Sort
nfile = Get number of strings
for i to nfile
    name$ = Get string: i
    basename$ = name$ - ".wav"
    Read from file: directory$+name$
    
    To Pitch (filtered ac): 0.0, 75, 600, 15, "off", 0.03, 0.09, 0.5, 0.055, 0.35, 0.14

    exists = 0
    selectObject: "Strings pitchlist"
    npitch = Get number of strings
    for k to npitch
        pitchname$ = Get string: k
        pitchbasename$ = pitchname$ - ".Pitch"
        if pitchbasename$ = basename$
            exists = 1
        endif
    endfor

    selectObject: "Pitch "+basename$
    Save as text file: outdirectory$+basename$+".Pitch"
    
    Smooth: smooth1
    Rename: "smooth"
    Interpolate
    Smooth: smooth2
    Down to PitchTier
    Save as headerless spreadsheet file: outdirectory$+basename$+".hesp"

    select all
    minusObject: "Strings list"
    minusObject: "Strings pitchlist"
    Remove
    selectObject: "Strings list"
endfor
Remove

