

#---------------------------------
# New invocation of recon-all Wed Jan  8 17:33:46 EET 2025 

 mri_convert /net/qnap/data/rd/ChildBrain/DATA/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/orig/001.mgz 

#--------------------------------------------
#@# MotionCor Wed Jan  8 17:33:54 EET 2025

 cp /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/orig/001.mgz /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/rawavg.mgz 


 mri_info /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/rawavg.mgz 


 mri_convert /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/rawavg.mgz /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/orig.mgz --conform 


 mri_add_xform_to_header -c /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/transforms/talairach.xfm /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/orig.mgz /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/orig.mgz 


 mri_info /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Wed Jan  8 17:34:01 EET 2025

 mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --ants-n4 --n 1 --proto-iters 1000 --distance 50 


 talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm 

talairach_avi log file is transforms/talairach_avi.log...

 cp transforms/talairach.auto.xfm transforms/talairach.xfm 

lta_convert --src orig.mgz --trg /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/mni305.cor.mgz --inxfm transforms/talairach.xfm --outlta transforms/talairach.xfm.lta --subject fsaverage --ltavox2vox
#--------------------------------------------
#@# Talairach Failure Detection Wed Jan  8 17:36:51 EET 2025

 talairach_afd -T 0.005 -xfm transforms/talairach.xfm 


 awk -f /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/bin/extract_talairach_avi_QA.awk /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/transforms/talairach_avi.log 


 tal_QC_AZS /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/transforms/talairach_avi.log 

#--------------------------------------------
#@# Nu Intensity Correction Wed Jan  8 17:36:51 EET 2025

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --n 2 --ants-n4 


 mri_add_xform_to_header -c /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Wed Jan  8 17:39:26 EET 2025

 mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz 

#--------------------------------------------
#@# Skull Stripping Wed Jan  8 17:40:48 EET 2025

 mri_em_register -skull nu.mgz /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/RB_all_withskull_2020_01_02.gca transforms/talairach_with_skull.lta 


 mri_watershed -T1 -brain_atlas /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/RB_all_withskull_2020_01_02.gca transforms/talairach_with_skull.lta T1.mgz brainmask.auto.mgz 


 cp brainmask.auto.mgz brainmask.mgz 

#-------------------------------------
#@# EM Registration Wed Jan  8 17:48:53 EET 2025

 mri_em_register -uns 3 -mask brainmask.mgz nu.mgz /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/RB_all_2020-01-02.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize Wed Jan  8 17:56:37 EET 2025

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/RB_all_2020-01-02.gca transforms/talairach.lta norm.mgz 

#--------------------------------------
#@# CA Reg Wed Jan  8 17:57:19 EET 2025

 mri_ca_register -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/RB_all_2020-01-02.gca transforms/talairach.m3z 

#--------------------------------------
#@# SubCort Seg Wed Jan  8 18:59:24 EET 2025

 mri_ca_label -relabel_unlikely 9 .3 -prior 0.5 -align norm.mgz transforms/talairach.m3z /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/RB_all_2020-01-02.gca aseg.auto_noCCseg.mgz 

#--------------------------------------
#@# CC Seg Wed Jan  8 19:37:45 EET 2025

 mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/transforms/cc_up.lta ICBM2009cNolinAsym 

#--------------------------------------
#@# Merge ASeg Wed Jan  8 19:38:09 EET 2025

 cp aseg.auto.mgz aseg.presurf.mgz 

#--------------------------------------------
#@# Intensity Normalization2 Wed Jan  8 19:38:09 EET 2025

 mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Wed Jan  8 19:40:35 EET 2025

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Wed Jan  8 19:40:37 EET 2025

 AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz 


 mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Wed Jan  8 19:42:49 EET 2025

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.presurf.mgz -ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/SubCorticalMassLUT.txt wm.mgz filled.mgz 

 cp filled.mgz filled.auto.mgz
#--------------------------------------------
#@# Tessellate lh Wed Jan  8 19:43:47 EET 2025

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Tessellate rh Wed Jan  8 19:43:51 EET 2025

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Wed Jan  8 19:43:55 EET 2025

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Smooth1 rh Wed Jan  8 19:43:57 EET 2025

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Wed Jan  8 19:43:59 EET 2025

 mris_inflate -no-save-sulc ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# Inflation1 rh Wed Jan  8 19:44:16 EET 2025

 mris_inflate -no-save-sulc ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Wed Jan  8 19:44:33 EET 2025

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# QSphere rh Wed Jan  8 19:46:32 EET 2025

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#@# Fix Topology lh Wed Jan  8 19:48:30 EET 2025

 mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 ICBM2009cNolinAsym lh 

#@# Fix Topology rh Wed Jan  8 19:50:53 EET 2025

 mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 ICBM2009cNolinAsym rh 


 mris_euler_number ../surf/lh.orig.premesh 


 mris_euler_number ../surf/rh.orig.premesh 


 mris_remesh --remesh --iters 3 --input /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.orig.premesh --output /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.orig 


 mris_remesh --remesh --iters 3 --input /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.orig.premesh --output /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.orig 


 mris_remove_intersection ../surf/lh.orig ../surf/lh.orig 


 rm -f ../surf/lh.inflated 


 mris_remove_intersection ../surf/rh.orig ../surf/rh.orig 


 rm -f ../surf/rh.inflated 

#--------------------------------------------
#@# AutoDetGWStats lh Wed Jan  8 19:55:34 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_autodet_gwstats --o ../surf/autodet.gw.stats.lh.dat --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/lh.orig.premesh
#--------------------------------------------
#@# AutoDetGWStats rh Wed Jan  8 19:55:38 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_autodet_gwstats --o ../surf/autodet.gw.stats.rh.dat --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/rh.orig.premesh
#--------------------------------------------
#@# WhitePreAparc lh Wed Jan  8 19:55:40 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --wm wm.mgz --threads 1 --invol brain.finalsurfs.mgz --lh --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white --seg aseg.presurf.mgz --nsmooth 5
#--------------------------------------------
#@# WhitePreAparc rh Wed Jan  8 20:00:09 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --wm wm.mgz --threads 1 --invol brain.finalsurfs.mgz --rh --i ../surf/rh.orig --o ../surf/rh.white.preaparc --white --seg aseg.presurf.mgz --nsmooth 5
#--------------------------------------------
#@# CortexLabel lh Wed Jan  8 20:05:50 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mri_label2label --label-cortex ../surf/lh.white.preaparc aseg.presurf.mgz 0 ../label/lh.cortex.label
#--------------------------------------------
#@# CortexLabel+HipAmyg lh Wed Jan  8 20:06:02 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mri_label2label --label-cortex ../surf/lh.white.preaparc aseg.presurf.mgz 1 ../label/lh.cortex+hipamyg.label
#--------------------------------------------
#@# CortexLabel rh Wed Jan  8 20:06:14 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mri_label2label --label-cortex ../surf/rh.white.preaparc aseg.presurf.mgz 0 ../label/rh.cortex.label
#--------------------------------------------
#@# CortexLabel+HipAmyg rh Wed Jan  8 20:06:26 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mri_label2label --label-cortex ../surf/rh.white.preaparc aseg.presurf.mgz 1 ../label/rh.cortex+hipamyg.label
#--------------------------------------------
#@# Smooth2 lh Wed Jan  8 20:06:38 EET 2025

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white.preaparc ../surf/lh.smoothwm 

#--------------------------------------------
#@# Smooth2 rh Wed Jan  8 20:06:41 EET 2025

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white.preaparc ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh Wed Jan  8 20:06:43 EET 2025

 mris_inflate ../surf/lh.smoothwm ../surf/lh.inflated 

#--------------------------------------------
#@# Inflation2 rh Wed Jan  8 20:07:05 EET 2025

 mris_inflate ../surf/rh.smoothwm ../surf/rh.inflated 

#--------------------------------------------
#@# Curv .H and .K lh Wed Jan  8 20:07:26 EET 2025

 mris_curvature -w -seed 1234 lh.white.preaparc 


 mris_curvature -seed 1234 -thresh .999 -n -a 5 -w -distances 10 10 lh.inflated 

#--------------------------------------------
#@# Curv .H and .K rh Wed Jan  8 20:08:23 EET 2025

 mris_curvature -w -seed 1234 rh.white.preaparc 


 mris_curvature -seed 1234 -thresh .999 -n -a 5 -w -distances 10 10 rh.inflated 

#--------------------------------------------
#@# Sphere lh Wed Jan  8 20:09:17 EET 2025

 mris_sphere -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Sphere rh Wed Jan  8 20:20:20 EET 2025

 mris_sphere -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 

#--------------------------------------------
#@# Surf Reg lh Wed Jan  8 20:38:24 EET 2025

 mris_register -curv ../surf/lh.sphere /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/lh.sphere.reg 


 ln -sf lh.sphere.reg lh.fsaverage.sphere.reg 

#--------------------------------------------
#@# Surf Reg rh Wed Jan  8 20:45:19 EET 2025

 mris_register -curv ../surf/rh.sphere /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/rh.sphere.reg 


 ln -sf rh.sphere.reg rh.fsaverage.sphere.reg 

#--------------------------------------------
#@# Jacobian white lh Wed Jan  8 20:57:06 EET 2025

 mris_jacobian ../surf/lh.white.preaparc ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# Jacobian white rh Wed Jan  8 20:57:08 EET 2025

 mris_jacobian ../surf/rh.white.preaparc ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

#--------------------------------------------
#@# AvgCurv lh Wed Jan  8 20:57:09 EET 2025

 mrisp_paint -a 5 /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/lh.sphere.reg ../surf/lh.avg_curv 

#--------------------------------------------
#@# AvgCurv rh Wed Jan  8 20:57:10 EET 2025

 mrisp_paint -a 5 /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/rh.sphere.reg ../surf/rh.avg_curv 

#-----------------------------------------
#@# Cortical Parc lh Wed Jan  8 20:57:10 EET 2025

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 ICBM2009cNolinAsym lh ../surf/lh.sphere.reg /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.annot 

#-----------------------------------------
#@# Cortical Parc rh Wed Jan  8 20:57:18 EET 2025

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 ICBM2009cNolinAsym rh ../surf/rh.sphere.reg /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.annot 

#--------------------------------------------
#@# WhiteSurfs lh Wed Jan  8 20:57:26 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 1 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white.preaparc --o ../surf/lh.white --white --nsmooth 0 --rip-label ../label/lh.cortex.label --rip-bg --rip-surf ../surf/lh.white.preaparc --aparc ../label/lh.aparc.annot
#--------------------------------------------
#@# WhiteSurfs rh Wed Jan  8 21:01:00 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --threads 1 --wm wm.mgz --invol brain.finalsurfs.mgz --rh --i ../surf/rh.white.preaparc --o ../surf/rh.white --white --nsmooth 0 --rip-label ../label/rh.cortex.label --rip-bg --rip-surf ../surf/rh.white.preaparc --aparc ../label/rh.aparc.annot
#--------------------------------------------
#@# T1PialSurf lh Wed Jan  8 21:05:15 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 1 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white --o ../surf/lh.pial.T1 --pial --nsmooth 0 --rip-label ../label/lh.cortex+hipamyg.label --pin-medial-wall ../label/lh.cortex.label --aparc ../label/lh.aparc.annot --repulse-surf ../surf/lh.white --white-surf ../surf/lh.white
#--------------------------------------------
#@# T1PialSurf rh Wed Jan  8 21:09:07 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --threads 1 --wm wm.mgz --invol brain.finalsurfs.mgz --rh --i ../surf/rh.white --o ../surf/rh.pial.T1 --pial --nsmooth 0 --rip-label ../label/rh.cortex+hipamyg.label --pin-medial-wall ../label/rh.cortex.label --aparc ../label/rh.aparc.annot --repulse-surf ../surf/rh.white --white-surf ../surf/rh.white
#@# white curv lh Wed Jan  8 21:12:47 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --curv-map ../surf/lh.white 2 10 ../surf/lh.curv
#@# white area lh Wed Jan  8 21:12:49 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --area-map ../surf/lh.white ../surf/lh.area
#@# pial curv lh Wed Jan  8 21:12:50 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --curv-map ../surf/lh.pial 2 10 ../surf/lh.curv.pial
#@# pial area lh Wed Jan  8 21:12:52 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --area-map ../surf/lh.pial ../surf/lh.area.pial
#@# thickness lh Wed Jan  8 21:12:53 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --thickness ../surf/lh.white ../surf/lh.pial 20 5 ../surf/lh.thickness
#@# area and vertex vol lh Wed Jan  8 21:13:32 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --thickness ../surf/lh.white ../surf/lh.pial 20 5 ../surf/lh.thickness
#@# white curv rh Wed Jan  8 21:13:33 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --curv-map ../surf/rh.white 2 10 ../surf/rh.curv
#@# white area rh Wed Jan  8 21:13:35 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --area-map ../surf/rh.white ../surf/rh.area
#@# pial curv rh Wed Jan  8 21:13:36 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --curv-map ../surf/rh.pial 2 10 ../surf/rh.curv.pial
#@# pial area rh Wed Jan  8 21:13:37 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --area-map ../surf/rh.pial ../surf/rh.area.pial
#@# thickness rh Wed Jan  8 21:13:38 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --thickness ../surf/rh.white ../surf/rh.pial 20 5 ../surf/rh.thickness
#@# area and vertex vol rh Wed Jan  8 21:14:07 EET 2025
cd /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri
mris_place_surface --thickness ../surf/rh.white ../surf/rh.pial 20 5 ../surf/rh.thickness

#-----------------------------------------
#@# Curvature Stats lh Wed Jan  8 21:14:09 EET 2025

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/lh.curv.stats -F smoothwm ICBM2009cNolinAsym lh curv sulc 


#-----------------------------------------
#@# Curvature Stats rh Wed Jan  8 21:14:11 EET 2025

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/rh.curv.stats -F smoothwm ICBM2009cNolinAsym rh curv sulc 

#--------------------------------------------
#@# Cortical ribbon mask Wed Jan  8 21:14:14 EET 2025

 mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon ICBM2009cNolinAsym 

#-----------------------------------------
#@# Cortical Parc 2 lh Wed Jan  8 21:19:07 EET 2025

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 ICBM2009cNolinAsym lh ../surf/lh.sphere.reg /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 2 rh Wed Jan  8 21:19:18 EET 2025

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 ICBM2009cNolinAsym rh ../surf/rh.sphere.reg /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 3 lh Wed Jan  8 21:19:29 EET 2025

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 ICBM2009cNolinAsym lh ../surf/lh.sphere.reg /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Cortical Parc 3 rh Wed Jan  8 21:19:38 EET 2025

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 ICBM2009cNolinAsym rh ../surf/rh.sphere.reg /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# WM/GM Contrast lh Wed Jan  8 21:19:45 EET 2025

 pctsurfcon --s ICBM2009cNolinAsym --lh-only 

#-----------------------------------------
#@# WM/GM Contrast rh Wed Jan  8 21:19:48 EET 2025

 pctsurfcon --s ICBM2009cNolinAsym --rh-only 

#-----------------------------------------
#@# Relabel Hypointensities Wed Jan  8 21:19:51 EET 2025

 mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz 

#-----------------------------------------
#@# APas-to-ASeg Wed Jan  8 21:20:06 EET 2025

 mri_surf2volseg --o aseg.mgz --i aseg.presurf.hypos.mgz --fix-presurf-with-ribbon /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/mri/ribbon.mgz --threads 1 --lh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.cortex.label --lh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.white --lh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.pial --rh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.cortex.label --rh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.white --rh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.pial 


 mri_brainvol_stats --subject ICBM2009cNolinAsym 

#-----------------------------------------
#@# AParc-to-ASeg aparc Wed Jan  8 21:20:15 EET 2025

 mri_surf2volseg --o aparc+aseg.mgz --label-cortex --i aseg.mgz --threads 1 --lh-annot /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.aparc.annot 1000 --lh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.cortex.label --lh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.white --lh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.pial --rh-annot /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.aparc.annot 2000 --rh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.cortex.label --rh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.white --rh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.pial 

#-----------------------------------------
#@# AParc-to-ASeg aparc.a2009s Wed Jan  8 21:22:37 EET 2025

 mri_surf2volseg --o aparc.a2009s+aseg.mgz --label-cortex --i aseg.mgz --threads 1 --lh-annot /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.aparc.a2009s.annot 11100 --lh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.cortex.label --lh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.white --lh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.pial --rh-annot /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.aparc.a2009s.annot 12100 --rh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.cortex.label --rh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.white --rh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.pial 

#-----------------------------------------
#@# AParc-to-ASeg aparc.DKTatlas Wed Jan  8 21:25:08 EET 2025

 mri_surf2volseg --o aparc.DKTatlas+aseg.mgz --label-cortex --i aseg.mgz --threads 1 --lh-annot /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.aparc.DKTatlas.annot 1000 --lh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.cortex.label --lh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.white --lh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.pial --rh-annot /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.aparc.DKTatlas.annot 2000 --rh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.cortex.label --rh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.white --rh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.pial 

#-----------------------------------------
#@# WMParc Wed Jan  8 21:27:37 EET 2025

 mri_surf2volseg --o wmparc.mgz --label-wm --i aparc+aseg.mgz --threads 1 --lh-annot /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.aparc.annot 3000 --lh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/lh.cortex.label --lh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.white --lh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/lh.pial --rh-annot /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.aparc.annot 4000 --rh-cortex-mask /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/label/rh.cortex.label --rh-white /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.white --rh-pial /home/amit3/DATA/FS_DIR/subjects/0FS741/ICBM2009cNolinAsym/surf/rh.pial 


 mri_segstats --seed 1234 --seg mri/wmparc.mgz --sum stats/wmparc.stats --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject ICBM2009cNolinAsym --surf-wm-vol --ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/WMParcStatsLUT.txt --etiv 

#-----------------------------------------
#@# Parcellation Stats lh Wed Jan  8 21:34:30 EET 2025

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab ICBM2009cNolinAsym lh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.pial.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab ICBM2009cNolinAsym lh pial 

#-----------------------------------------
#@# Parcellation Stats rh Wed Jan  8 21:34:58 EET 2025

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab ICBM2009cNolinAsym rh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.pial.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab ICBM2009cNolinAsym rh pial 

#-----------------------------------------
#@# Parcellation Stats 2 lh Wed Jan  8 21:35:26 EET 2025

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.a2009s.stats -b -a ../label/lh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab ICBM2009cNolinAsym lh white 

#-----------------------------------------
#@# Parcellation Stats 2 rh Wed Jan  8 21:35:41 EET 2025

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.a2009s.stats -b -a ../label/rh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab ICBM2009cNolinAsym rh white 

#-----------------------------------------
#@# Parcellation Stats 3 lh Wed Jan  8 21:35:55 EET 2025

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.DKTatlas.stats -b -a ../label/lh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab ICBM2009cNolinAsym lh white 

#-----------------------------------------
#@# Parcellation Stats 3 rh Wed Jan  8 21:36:08 EET 2025

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.DKTatlas.stats -b -a ../label/rh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab ICBM2009cNolinAsym rh white 

#--------------------------------------------
#@# ASeg Stats Wed Jan  8 21:36:21 EET 2025

 mri_segstats --seed 1234 --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/ASegStatsLUT.txt --subject ICBM2009cNolinAsym 

INFO: fsaverage subject does not exist in SUBJECTS_DIR
INFO: Creating symlink to fsaverage subject...

 cd /home/amit3/DATA/FS_DIR/subjects/0FS741; ln -s /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/subjects/fsaverage; cd - 

#--------------------------------------------
#@# BA_exvivo Labels lh Wed Jan  8 21:39:35 EET 2025

 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA1_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA2_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA3a_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA3a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA3b_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA3b_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA4a_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA4a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA4p_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA4p_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA6_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA6_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA44_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA44_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA45_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA45_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.V1_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.V1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.V2_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.V2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.MT_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.MT_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.entorhinal_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.entorhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.perirhinal_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.perirhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.FG1.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.FG1.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.FG2.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.FG2.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.FG3.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.FG3.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.FG4.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.FG4.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.hOc1.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.hOc1.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.hOc2.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.hOc2.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.hOc3v.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.hOc3v.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.hOc4v.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.hOc4v.mpm.vpnl.label --hemi lh --regmethod surface 


 mris_label2annot --s ICBM2009cNolinAsym --ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/colortable_vpnl.txt --hemi lh --a mpm.vpnl --maxstatwinner --noverbose --l lh.FG1.mpm.vpnl.label --l lh.FG2.mpm.vpnl.label --l lh.FG3.mpm.vpnl.label --l lh.FG4.mpm.vpnl.label --l lh.hOc1.mpm.vpnl.label --l lh.hOc2.mpm.vpnl.label --l lh.hOc3v.mpm.vpnl.label --l lh.hOc4v.mpm.vpnl.label 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA1_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA2_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA3a_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA3a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA3b_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA3b_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA4a_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA4a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA4p_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA4p_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA6_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA6_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA44_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA44_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.BA45_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.BA45_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.V1_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.V1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.V2_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.V2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.MT_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.MT_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.entorhinal_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.entorhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/lh.perirhinal_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./lh.perirhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mris_label2annot --s ICBM2009cNolinAsym --hemi lh --ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/colortable_BA.txt --l lh.BA1_exvivo.label --l lh.BA2_exvivo.label --l lh.BA3a_exvivo.label --l lh.BA3b_exvivo.label --l lh.BA4a_exvivo.label --l lh.BA4p_exvivo.label --l lh.BA6_exvivo.label --l lh.BA44_exvivo.label --l lh.BA45_exvivo.label --l lh.V1_exvivo.label --l lh.V2_exvivo.label --l lh.MT_exvivo.label --l lh.perirhinal_exvivo.label --l lh.entorhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s ICBM2009cNolinAsym --hemi lh --ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/colortable_BA.txt --l lh.BA1_exvivo.thresh.label --l lh.BA2_exvivo.thresh.label --l lh.BA3a_exvivo.thresh.label --l lh.BA3b_exvivo.thresh.label --l lh.BA4a_exvivo.thresh.label --l lh.BA4p_exvivo.thresh.label --l lh.BA6_exvivo.thresh.label --l lh.BA44_exvivo.thresh.label --l lh.BA45_exvivo.thresh.label --l lh.V1_exvivo.thresh.label --l lh.V2_exvivo.thresh.label --l lh.MT_exvivo.thresh.label --l lh.perirhinal_exvivo.thresh.label --l lh.entorhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.stats -b -a ./lh.BA_exvivo.annot -c ./BA_exvivo.ctab ICBM2009cNolinAsym lh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.thresh.stats -b -a ./lh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab ICBM2009cNolinAsym lh white 

#--------------------------------------------
#@# BA_exvivo Labels rh Wed Jan  8 21:42:38 EET 2025

 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA1_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA2_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA3a_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA3a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA3b_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA3b_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA4a_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA4a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA4p_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA4p_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA6_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA6_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA44_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA44_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA45_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA45_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.V1_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.V1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.V2_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.V2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.MT_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.MT_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.entorhinal_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.entorhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.perirhinal_exvivo.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.perirhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.FG1.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.FG1.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.FG2.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.FG2.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.FG3.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.FG3.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.FG4.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.FG4.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.hOc1.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.hOc1.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.hOc2.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.hOc2.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.hOc3v.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.hOc3v.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.hOc4v.mpm.vpnl.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.hOc4v.mpm.vpnl.label --hemi rh --regmethod surface 


 mris_label2annot --s ICBM2009cNolinAsym --ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/colortable_vpnl.txt --hemi rh --a mpm.vpnl --maxstatwinner --noverbose --l rh.FG1.mpm.vpnl.label --l rh.FG2.mpm.vpnl.label --l rh.FG3.mpm.vpnl.label --l rh.FG4.mpm.vpnl.label --l rh.hOc1.mpm.vpnl.label --l rh.hOc2.mpm.vpnl.label --l rh.hOc3v.mpm.vpnl.label --l rh.hOc4v.mpm.vpnl.label 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA1_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA2_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA3a_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA3a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA3b_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA3b_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA4a_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA4a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA4p_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA4p_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA6_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA6_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA44_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA44_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.BA45_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.BA45_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.V1_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.V1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.V2_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.V2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.MT_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.MT_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.entorhinal_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.entorhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/amit3/DATA/FS_DIR/subjects/0FS741/fsaverage/label/rh.perirhinal_exvivo.thresh.label --trgsubject ICBM2009cNolinAsym --trglabel ./rh.perirhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mris_label2annot --s ICBM2009cNolinAsym --hemi rh --ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/colortable_BA.txt --l rh.BA1_exvivo.label --l rh.BA2_exvivo.label --l rh.BA3a_exvivo.label --l rh.BA3b_exvivo.label --l rh.BA4a_exvivo.label --l rh.BA4p_exvivo.label --l rh.BA6_exvivo.label --l rh.BA44_exvivo.label --l rh.BA45_exvivo.label --l rh.V1_exvivo.label --l rh.V2_exvivo.label --l rh.MT_exvivo.label --l rh.perirhinal_exvivo.label --l rh.entorhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s ICBM2009cNolinAsym --hemi rh --ctab /home/amit3/0Software/FreeSurf/FS741/usr/local/freesurfer/7.4.1/average/colortable_BA.txt --l rh.BA1_exvivo.thresh.label --l rh.BA2_exvivo.thresh.label --l rh.BA3a_exvivo.thresh.label --l rh.BA3b_exvivo.thresh.label --l rh.BA4a_exvivo.thresh.label --l rh.BA4p_exvivo.thresh.label --l rh.BA6_exvivo.thresh.label --l rh.BA44_exvivo.thresh.label --l rh.BA45_exvivo.thresh.label --l rh.V1_exvivo.thresh.label --l rh.V2_exvivo.thresh.label --l rh.MT_exvivo.thresh.label --l rh.perirhinal_exvivo.thresh.label --l rh.entorhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.stats -b -a ./rh.BA_exvivo.annot -c ./BA_exvivo.ctab ICBM2009cNolinAsym rh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.thresh.stats -b -a ./rh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab ICBM2009cNolinAsym rh white 

