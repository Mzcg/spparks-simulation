echo "Start Program : $(date)" >>logAllruns.log
mpirun -np 28 "/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/spk_mpi" < "/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/generated_SPPARKS_scripts_cannotsee/speed_3_mpwidth_69_haz_114.0_thickness_10.0/3D_AMsim_speed_3_mpwidth_69_haz_114.0_thickness_10.0.in"
mv ./*.dump "/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/generated_SPPARKS_scripts_cannotsee/speed_3_mpwidth_69_haz_114.0_thickness_10.0"
mv ./*.spparks "/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/generated_SPPARKS_scripts_cannotsee/speed_3_mpwidth_69_haz_114.0_thickness_10.0"
mv ./*.variable "/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/generated_SPPARKS_scripts_cannotsee/speed_3_mpwidth_69_haz_114.0_thickness_10.0"
echo "prgram ran for folder speed_3_mpwidth_69_haz_114.0_thickness_10.0" >>logAllruns.log 
echo "End Program: $(date)" >> logAllruns.log