::  Move files from folders and subfolders to a different one, useful to put nested files in a single place
::  Source: https://superuser.com/questions/999922/move-all-files-from-multiple-subfolders-into-the-parent-folder

FOR /R "C:\Source Folder" %i IN (*.*) DO MOVE "%i" "C:\Staging Folder"
