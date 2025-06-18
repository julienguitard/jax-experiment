#!/usr/bin/env zsh

cd '{persistent_data_path}'
rm -f '{file_name}' 
curl -o '{file_name}' '''{data_source}'''

