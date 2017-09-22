# pymake Command Line


```bash
init = command$;
command = 'pmk' command_name expedesign_id [expe_id]* [pmk_options];
command_name = 'run' | 'runpara' | 'path' | 'cmd' | 'update' | 'show' | 'hist' |  '' ;
expe_id = int; # unique int identifier of an expe from 0 to; size(exp) -1.
expedesign_id = [experience id/name]; # string identifier to an exp
pmk_options = [pymake and global options];
```

### Command_name 
(alors si cest vide ici, ca va aller chercher lexperience par default, qui doit être def avant l' init du script. Pour voir les scripts "pmk -l script";
if empty, a defaut settings a taken from {_default_expe} define as a constant in a script (ExpeFormat). To list the script `pmk -l script`

Remark : -l and -s (--simulate) options don't execute, they just show things up.

### Expedesign_id
se sont toutes les expériences qui sont dans expedesign/, pour les voir tu peux faire "pmk -l expe" (ou juste" pmk -l")
Pick among all (design of )experiences in {spec}. To list thel `pmk -l spec` (or juste `pmk -l`)

### pmk_options
Here are all the special options that own pymake, such as --refdir, --format, --script, -w, -l, -h etc. Additionally, all the options for the courant project sould be added in the `grammarg.py` file.
