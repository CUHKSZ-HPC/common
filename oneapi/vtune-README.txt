1. source variables: . oneapi.sh
2. collect data: vtune -collect hpc-performance -r <result_dir_path> -- <application_path>
3. visualize data: vtune-backend --data-directory=<parent_of_result_dir> --allow-remote-access

FAQ:
1. Will `source` change defalt MPI? Yes.
2. What other statstics are available? Check cheatsheet.
3. In visualize data, I need to set a password, will that interfere other people's settings? No. Their are user dependent.
4. Why there's a `intel` folder appeared after using vtune. They just created by vtune. You can safely delete it. But if you use vtune again, it will appear.

