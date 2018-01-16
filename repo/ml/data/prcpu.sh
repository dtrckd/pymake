#!/bin/bash

ps -u adulac   -C python,parallel  -o user,pid,cmd=CMD 
