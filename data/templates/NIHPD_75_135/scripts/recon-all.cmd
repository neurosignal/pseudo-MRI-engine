

#---------------------------------
# New invocation of recon-all Mon May 11 00:10:17 EEST 2026 
#--------------------------------------------
#@# T2/FLAIR Input Mon May 11 00:10:18 EEST 2026

 mri_convert --no_scale 1 /home/neurosign/Downloads/nihpd_asym_07.5-13.5_nifti/nihpd_asym_07.5-13.5_t2w.nii /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/orig/T2raw.mgz 

#--------------------------------------------
#@# MotionCor Mon May 11 00:10:19 EEST 2026

 cp /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/orig/001.mgz /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/rawavg.mgz 


 mri_info /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/rawavg.mgz 


 mri_convert /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/rawavg.mgz /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/orig.mgz --conform_min 


 mri_add_xform_to_header -c /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/transforms/talairach.xfm /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/orig.mgz /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/orig.mgz 


 mri_info /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Mon May 11 00:10:23 EEST 2026

 mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --ants-n4 --n 1 --proto-iters 1000 --distance 50 


 talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm 

talairach_avi log file is transforms/talairach_avi.log...

 cp transforms/talairach.auto.xfm transforms/talairach.xfm 

lta_convert --src orig.mgz --trg /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/mni305.cor.mgz --inxfm transforms/talairach.xfm --outlta transforms/talairach.xfm.lta --subject fsaverage --ltavox2vox
#--------------------------------------------
#@# Talairach Failure Detection Mon May 11 00:13:16 EEST 2026

 talairach_afd -T 0.005 -xfm transforms/talairach.xfm 


 awk -f /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/bin/extract_talairach_avi_QA.awk /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/transforms/talairach_avi.log 


 tal_QC_AZS /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/transforms/talairach_avi.log 

#--------------------------------------------
#@# Nu Intensity Correction Mon May 11 00:13:16 EEST 2026

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --cm --n 2 --ants-n4 


 mri_add_xform_to_header -c /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Mon May 11 00:15:48 EEST 2026

 mri_normalize -g 1 -seed 1234 -mprage -noconform nu.mgz T1.mgz 

#--------------------------------------------
#@# Skull Stripping Mon May 11 00:16:43 EEST 2026

 mri_em_register -skull nu.mgz /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/RB_all_withskull_2020_01_02.gca transforms/talairach_with_skull.lta 


 mri_watershed -T1 -brain_atlas /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/RB_all_withskull_2020_01_02.gca transforms/talairach_with_skull.lta -atlas T1.mgz brainmask.auto.mgz 


INFO: brainmask.mgz already exists!
It will not be overwritten.
This is done to retain any edits made to brainmask.mgz.
Add the -clean-bm flag to recon-all to overwrite brainmask.mgz.

#-------------------------------------
#@# EM Registration Mon May 11 00:18:25 EEST 2026

 mri_em_register -uns 3 -mask brainmask.mgz nu.mgz /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize Mon May 11 00:19:59 EEST 2026

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.lta norm.mgz 

#--------------------------------------
#@# CA Reg Mon May 11 00:20:34 EEST 2026

 mri_ca_register -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.m3z 

#--------------------------------------
#@# SubCort Seg Mon May 11 07:24:13 EEST 2026

 mri_ca_label -relabel_unlikely 9 .3 -prior 0.5 -align norm.mgz transforms/talairach.m3z /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/RB_all_2020-01-02.gca aseg.auto_noCCseg.mgz 

#--------------------------------------
#@# CC Seg Mon May 11 10:56:44 EEST 2026

 mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/transforms/cc_up.lta NIHPD_75_135 

#--------------------------------------
#@# Merge ASeg Mon May 11 10:57:03 EEST 2026

 cp aseg.auto.mgz aseg.presurf.mgz 

#--------------------------------------------
#@# Intensity Normalization2 Mon May 11 10:57:03 EEST 2026

 mri_normalize -seed 1234 -mprage -noconform -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Mon May 11 10:58:33 EEST 2026

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Mon May 11 10:58:34 EEST 2026

 AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz 


 mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Mon May 11 11:00:28 EEST 2026

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.presurf.mgz -ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/SubCorticalMassLUT.txt wm.mgz filled.mgz 

 cp filled.mgz filled.auto.mgz
#--------------------------------------------
#@# Tessellate lh Mon May 11 11:01:15 EEST 2026

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix.predec 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix.predec ../surf/lh.orig.nofix.predec 


 mris_remesh --desired-face-area 0.5 --input ../surf/lh.orig.nofix.predec --output ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Tessellate rh Mon May 11 11:01:25 EEST 2026

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix.predec 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix.predec ../surf/rh.orig.nofix.predec 


 mris_remesh --desired-face-area 0.5 --input ../surf/rh.orig.nofix.predec --output ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Mon May 11 11:01:35 EEST 2026

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Smooth1 rh Mon May 11 11:01:37 EEST 2026

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Mon May 11 11:01:39 EEST 2026

 mris_inflate -no-save-sulc ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# Inflation1 rh Mon May 11 11:01:50 EEST 2026

 mris_inflate -no-save-sulc ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Mon May 11 11:02:01 EEST 2026

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# QSphere rh Mon May 11 11:03:17 EEST 2026

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#@# Fix Topology lh Mon May 11 11:04:32 EEST 2026

 mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 NIHPD_75_135 lh 

#@# Fix Topology rh Mon May 11 11:05:20 EEST 2026

 mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 NIHPD_75_135 rh 


 mris_euler_number ../surf/lh.orig.premesh 


 mris_euler_number ../surf/rh.orig.premesh 


 mris_remesh --remesh --iters 3 --input /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.orig.premesh --output /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.orig 


 mris_remesh --remesh --iters 3 --input /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.orig.premesh --output /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.orig 


 mris_remove_intersection ../surf/lh.orig ../surf/lh.orig 


 rm -f ../surf/lh.inflated 


 mris_remove_intersection ../surf/rh.orig ../surf/rh.orig 


 rm -f ../surf/rh.inflated 

#--------------------------------------------
#@# AutoDetGWStats lh Mon May 11 11:07:30 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_autodet_gwstats --o ../surf/autodet.gw.stats.lh.dat --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/lh.orig.premesh
#--------------------------------------------
#@# AutoDetGWStats rh Mon May 11 11:07:33 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_autodet_gwstats --o ../surf/autodet.gw.stats.rh.dat --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/rh.orig.premesh
#--------------------------------------------
#@# WhitePreAparc lh Mon May 11 11:07:35 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --wm wm.mgz --threads 8 --invol brain.finalsurfs.mgz --lh --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white --seg aseg.presurf.mgz --nsmooth 5
#--------------------------------------------
#@# WhitePreAparc rh Mon May 11 11:10:07 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --wm wm.mgz --threads 8 --invol brain.finalsurfs.mgz --rh --i ../surf/rh.orig --o ../surf/rh.white.preaparc --white --seg aseg.presurf.mgz --nsmooth 5
#--------------------------------------------
#@# CortexLabel lh Mon May 11 11:12:21 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mri_label2label --label-cortex ../surf/lh.white.preaparc aseg.presurf.mgz 0 ../label/lh.cortex.label
#--------------------------------------------
#@# CortexLabel+HipAmyg lh Mon May 11 11:12:30 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mri_label2label --label-cortex ../surf/lh.white.preaparc aseg.presurf.mgz 1 ../label/lh.cortex+hipamyg.label
#--------------------------------------------
#@# CortexLabel rh Mon May 11 11:12:39 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mri_label2label --label-cortex ../surf/rh.white.preaparc aseg.presurf.mgz 0 ../label/rh.cortex.label
#--------------------------------------------
#@# CortexLabel+HipAmyg rh Mon May 11 11:12:48 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mri_label2label --label-cortex ../surf/rh.white.preaparc aseg.presurf.mgz 1 ../label/rh.cortex+hipamyg.label
#--------------------------------------------
#@# Smooth2 lh Mon May 11 11:12:57 EEST 2026

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white.preaparc ../surf/lh.smoothwm 

#--------------------------------------------
#@# Smooth2 rh Mon May 11 11:13:00 EEST 2026

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white.preaparc ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh Mon May 11 11:13:02 EEST 2026

 mris_inflate ../surf/lh.smoothwm ../surf/lh.inflated 

#--------------------------------------------
#@# Inflation2 rh Mon May 11 11:13:14 EEST 2026

 mris_inflate ../surf/rh.smoothwm ../surf/rh.inflated 

#--------------------------------------------
#@# Curv .H and .K lh Mon May 11 11:13:26 EEST 2026

 mris_curvature -w -seed 1234 lh.white.preaparc 


 mris_curvature -seed 1234 -thresh .999 -n -a 5 -w -distances 10 10 lh.inflated 

#--------------------------------------------
#@# Curv .H and .K rh Mon May 11 11:14:09 EEST 2026

 mris_curvature -w -seed 1234 rh.white.preaparc 


 mris_curvature -seed 1234 -thresh .999 -n -a 5 -w -distances 10 10 rh.inflated 

#--------------------------------------------
#@# Sphere lh Mon May 11 11:14:52 EEST 2026

 mris_sphere -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Sphere rh Mon May 11 11:17:02 EEST 2026

 mris_sphere -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 

#--------------------------------------------
#@# Surf Reg lh Mon May 11 11:19:15 EEST 2026

 mris_register -curv ../surf/lh.sphere /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/lh.sphere.reg 


 ln -sf lh.sphere.reg lh.fsaverage.sphere.reg 

#--------------------------------------------
#@# Surf Reg rh Mon May 11 11:22:36 EEST 2026

 mris_register -curv ../surf/rh.sphere /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/rh.sphere.reg 


 ln -sf rh.sphere.reg rh.fsaverage.sphere.reg 

#--------------------------------------------
#@# Jacobian white lh Mon May 11 11:25:49 EEST 2026

 mris_jacobian ../surf/lh.white.preaparc ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# Jacobian white rh Mon May 11 11:25:49 EEST 2026

 mris_jacobian ../surf/rh.white.preaparc ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

#--------------------------------------------
#@# AvgCurv lh Mon May 11 11:25:50 EEST 2026

 mrisp_paint -a 5 /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/lh.sphere.reg ../surf/lh.avg_curv 

#--------------------------------------------
#@# AvgCurv rh Mon May 11 11:25:51 EEST 2026

 mrisp_paint -a 5 /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/rh.sphere.reg ../surf/rh.avg_curv 

#-----------------------------------------
#@# Cortical Parc lh Mon May 11 11:25:52 EEST 2026

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 NIHPD_75_135 lh ../surf/lh.sphere.reg /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.annot 

#-----------------------------------------
#@# Cortical Parc rh Mon May 11 11:26:00 EEST 2026

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 NIHPD_75_135 rh ../surf/rh.sphere.reg /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.annot 

#--------------------------------------------
#@# WhiteSurfs lh Mon May 11 11:26:09 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 8 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white.preaparc --o ../surf/lh.white --white --nsmooth 0 --rip-label ../label/lh.cortex.label --rip-bg --rip-surf ../surf/lh.white.preaparc --aparc ../label/lh.aparc.annot
#--------------------------------------------
#@# WhiteSurfs rh Mon May 11 11:28:33 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --threads 8 --wm wm.mgz --invol brain.finalsurfs.mgz --rh --i ../surf/rh.white.preaparc --o ../surf/rh.white --white --nsmooth 0 --rip-label ../label/rh.cortex.label --rip-bg --rip-surf ../surf/rh.white.preaparc --aparc ../label/rh.aparc.annot
#--------------------------------------------
#@# T1PialSurf lh Mon May 11 11:30:57 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 8 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white --o ../surf/lh.pial.T1 --pial --nsmooth 0 --rip-label ../label/lh.cortex+hipamyg.label --pin-medial-wall ../label/lh.cortex.label --aparc ../label/lh.aparc.annot --repulse-surf ../surf/lh.white --white-surf ../surf/lh.white
#--------------------------------------------
#@# T1PialSurf rh Mon May 11 11:33:47 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --threads 8 --wm wm.mgz --invol brain.finalsurfs.mgz --rh --i ../surf/rh.white --o ../surf/rh.pial.T1 --pial --nsmooth 0 --rip-label ../label/rh.cortex+hipamyg.label --pin-medial-wall ../label/rh.cortex.label --aparc ../label/rh.aparc.annot --repulse-surf ../surf/rh.white --white-surf ../surf/rh.white
#--------------------------------------------
#@# Refine Pial Surfs w/ T2/FLAIR Mon May 11 11:36:39 EEST 2026

 bbregister --s NIHPD_75_135 --mov /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/orig/T2raw.mgz --lta /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/transforms/T2raw.auto.lta --init-coreg --T2 --gm-proj-abs 2 --wm-proj-abs 1 --no-coreg-ref-mask 


 cp /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/transforms/T2raw.auto.lta /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/transforms/T2raw.lta 


 mri_normalize -seed 1234 -sigma 0.5 -nonmax_suppress 0 -min_dist 1 -aseg /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/aseg.presurf.mgz -surface /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.white identity.nofile -surface /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.white identity.nofile /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/T2.prenorm.mgz /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/T2.norm.mgz 


 mri_mask -transfer 255 -keep_mask_deletion_edits /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/T2.norm.mgz /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/brain.finalsurfs.mgz /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/T2.mgz 

#--------------------------------------------
#@# MMPialSurf lh Mon May 11 11:38:17 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --wm wm.mgz --threads 8 --invol brain.finalsurfs.mgz --lh --i ../surf/lh.pial.T1 --o ../surf/lh.pial.T2 --pial --nsmooth 0 --rip-label ../label/lh.cortex+hipamyg.label --pin-medial-wall ../label/lh.cortex.label --white-surf ../surf/lh.white --aparc ../label/lh.aparc.annot --repulse-surf ../surf/lh.white --mmvol T2.mgz T2
#--------------------------------------------
#@# MMPialSurf rh Mon May 11 11:47:59 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --wm wm.mgz --threads 8 --invol brain.finalsurfs.mgz --rh --i ../surf/rh.pial.T1 --o ../surf/rh.pial.T2 --pial --nsmooth 0 --rip-label ../label/rh.cortex+hipamyg.label --pin-medial-wall ../label/rh.cortex.label --white-surf ../surf/rh.white --aparc ../label/rh.aparc.annot --repulse-surf ../surf/rh.white --mmvol T2.mgz T2
#@# white curv lh Mon May 11 11:56:50 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --curv-map ../surf/lh.white 2 10 ../surf/lh.curv
#@# white area lh Mon May 11 11:56:51 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --area-map ../surf/lh.white ../surf/lh.area
#@# pial curv lh Mon May 11 11:56:52 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --curv-map ../surf/lh.pial 2 10 ../surf/lh.curv.pial
#@# pial area lh Mon May 11 11:56:53 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --area-map ../surf/lh.pial ../surf/lh.area.pial
#@# thickness lh Mon May 11 11:56:54 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --thickness ../surf/lh.white ../surf/lh.pial 20 5 ../surf/lh.thickness
#@# area and vertex vol lh Mon May 11 11:57:47 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --thickness ../surf/lh.white ../surf/lh.pial 20 5 ../surf/lh.thickness
#@# white curv rh Mon May 11 11:57:48 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --curv-map ../surf/rh.white 2 10 ../surf/rh.curv
#@# white area rh Mon May 11 11:57:50 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --area-map ../surf/rh.white ../surf/rh.area
#@# pial curv rh Mon May 11 11:57:50 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --curv-map ../surf/rh.pial 2 10 ../surf/rh.curv.pial
#@# pial area rh Mon May 11 11:57:52 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --area-map ../surf/rh.pial ../surf/rh.area.pial
#@# thickness rh Mon May 11 11:57:52 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --thickness ../surf/rh.white ../surf/rh.pial 20 5 ../surf/rh.thickness
#@# area and vertex vol rh Mon May 11 11:58:23 EEST 2026
cd /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri
mris_place_surface --thickness ../surf/rh.white ../surf/rh.pial 20 5 ../surf/rh.thickness

#-----------------------------------------
#@# Curvature Stats lh Mon May 11 11:58:24 EEST 2026

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/lh.curv.stats -F smoothwm NIHPD_75_135 lh curv sulc 


#-----------------------------------------
#@# Curvature Stats rh Mon May 11 11:58:26 EEST 2026

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/rh.curv.stats -F smoothwm NIHPD_75_135 rh curv sulc 

#--------------------------------------------
#@# Cortical ribbon mask Mon May 11 11:58:27 EEST 2026

 mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon NIHPD_75_135 

#-----------------------------------------
#@# Cortical Parc 2 lh Mon May 11 12:05:32 EEST 2026

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 NIHPD_75_135 lh ../surf/lh.sphere.reg /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 2 rh Mon May 11 12:05:43 EEST 2026

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 NIHPD_75_135 rh ../surf/rh.sphere.reg /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 3 lh Mon May 11 12:05:55 EEST 2026

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 NIHPD_75_135 lh ../surf/lh.sphere.reg /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Cortical Parc 3 rh Mon May 11 12:06:03 EEST 2026

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 NIHPD_75_135 rh ../surf/rh.sphere.reg /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# WM/GM Contrast lh Mon May 11 12:06:12 EEST 2026

 pctsurfcon --s NIHPD_75_135 --lh-only 

#-----------------------------------------
#@# WM/GM Contrast rh Mon May 11 12:06:15 EEST 2026

 pctsurfcon --s NIHPD_75_135 --rh-only 

#-----------------------------------------
#@# Relabel Hypointensities Mon May 11 12:06:18 EEST 2026

 mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz 

#-----------------------------------------
#@# APas-to-ASeg Mon May 11 12:06:32 EEST 2026

 mri_surf2volseg --o aseg.mgz --i aseg.presurf.hypos.mgz --fix-presurf-with-ribbon /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/mri/ribbon.mgz --threads 8 --lh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.cortex.label --lh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.white --lh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.pial --rh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.cortex.label --rh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.white --rh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.pial 


 mri_brainvol_stats --subject NIHPD_75_135 

#-----------------------------------------
#@# AParc-to-ASeg aparc Mon May 11 12:06:40 EEST 2026

 mri_surf2volseg --o aparc+aseg.mgz --label-cortex --i aseg.mgz --threads 8 --lh-annot /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.aparc.annot 1000 --lh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.cortex.label --lh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.white --lh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.pial --rh-annot /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.aparc.annot 2000 --rh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.cortex.label --rh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.white --rh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.pial 

#-----------------------------------------
#@# AParc-to-ASeg aparc.a2009s Mon May 11 12:07:36 EEST 2026

 mri_surf2volseg --o aparc.a2009s+aseg.mgz --label-cortex --i aseg.mgz --threads 8 --lh-annot /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.aparc.a2009s.annot 11100 --lh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.cortex.label --lh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.white --lh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.pial --rh-annot /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.aparc.a2009s.annot 12100 --rh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.cortex.label --rh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.white --rh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.pial 

#-----------------------------------------
#@# AParc-to-ASeg aparc.DKTatlas Mon May 11 12:08:33 EEST 2026

 mri_surf2volseg --o aparc.DKTatlas+aseg.mgz --label-cortex --i aseg.mgz --threads 8 --lh-annot /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.aparc.DKTatlas.annot 1000 --lh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.cortex.label --lh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.white --lh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.pial --rh-annot /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.aparc.DKTatlas.annot 2000 --rh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.cortex.label --rh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.white --rh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.pial 

#-----------------------------------------
#@# WMParc Mon May 11 12:09:34 EEST 2026

 mri_surf2volseg --o wmparc.mgz --label-wm --i aparc+aseg.mgz --threads 8 --lh-annot /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.aparc.annot 3000 --lh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/lh.cortex.label --lh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.white --lh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/lh.pial --rh-annot /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.aparc.annot 4000 --rh-cortex-mask /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/label/rh.cortex.label --rh-white /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.white --rh-pial /home/neurosign/FS_SUBS_DIR/fs741/NIHPD_75_135/surf/rh.pial 


 mri_segstats --seed 1234 --seg mri/wmparc.mgz --sum stats/wmparc.stats --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject NIHPD_75_135 --surf-wm-vol --ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/WMParcStatsLUT.txt --etiv 

#-----------------------------------------
#@# Parcellation Stats lh Mon May 11 12:13:13 EEST 2026

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab NIHPD_75_135 lh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.pial.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab NIHPD_75_135 lh pial 

#-----------------------------------------
#@# Parcellation Stats rh Mon May 11 12:13:34 EEST 2026

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab NIHPD_75_135 rh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.pial.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab NIHPD_75_135 rh pial 

#-----------------------------------------
#@# Parcellation Stats 2 lh Mon May 11 12:13:57 EEST 2026

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.a2009s.stats -b -a ../label/lh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab NIHPD_75_135 lh white 

#-----------------------------------------
#@# Parcellation Stats 2 rh Mon May 11 12:14:08 EEST 2026

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.a2009s.stats -b -a ../label/rh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab NIHPD_75_135 rh white 

#-----------------------------------------
#@# Parcellation Stats 3 lh Mon May 11 12:14:19 EEST 2026

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.DKTatlas.stats -b -a ../label/lh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab NIHPD_75_135 lh white 

#-----------------------------------------
#@# Parcellation Stats 3 rh Mon May 11 12:14:30 EEST 2026

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.DKTatlas.stats -b -a ../label/rh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab NIHPD_75_135 rh white 

#--------------------------------------------
#@# ASeg Stats Mon May 11 12:14:41 EEST 2026

 mri_segstats --seed 1234 --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/ASegStatsLUT.txt --subject NIHPD_75_135 

INFO: fsaverage subject does not exist in SUBJECTS_DIR
INFO: Creating symlink to fsaverage subject...

 cd /home/neurosign/FS_SUBS_DIR/fs741; ln -s /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/subjects/fsaverage; cd - 

#--------------------------------------------
#@# BA_exvivo Labels lh Mon May 11 12:15:22 EEST 2026

 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA1_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA2_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA3a_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA3a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA3b_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA3b_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA4a_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA4a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA4p_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA4p_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA6_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA6_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA44_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA44_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA45_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA45_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.V1_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.V1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.V2_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.V2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.MT_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.MT_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.entorhinal_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.entorhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.perirhinal_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./lh.perirhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.FG1.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./lh.FG1.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.FG2.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./lh.FG2.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.FG3.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./lh.FG3.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.FG4.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./lh.FG4.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.hOc1.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./lh.hOc1.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.hOc2.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./lh.hOc2.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.hOc3v.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./lh.hOc3v.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.hOc4v.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./lh.hOc4v.mpm.vpnl.label --hemi lh --regmethod surface 


 mris_label2annot --s NIHPD_75_135 --ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/colortable_vpnl.txt --hemi lh --a mpm.vpnl --maxstatwinner --noverbose --l lh.FG1.mpm.vpnl.label --l lh.FG2.mpm.vpnl.label --l lh.FG3.mpm.vpnl.label --l lh.FG4.mpm.vpnl.label --l lh.hOc1.mpm.vpnl.label --l lh.hOc2.mpm.vpnl.label --l lh.hOc3v.mpm.vpnl.label --l lh.hOc4v.mpm.vpnl.label 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA1_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA2_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA3a_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA3a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA3b_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA3b_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA4a_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA4a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA4p_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA4p_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA6_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA6_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA44_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA44_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.BA45_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.BA45_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.V1_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.V1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.V2_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.V2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.MT_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.MT_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.entorhinal_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.entorhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/lh.perirhinal_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./lh.perirhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mris_label2annot --s NIHPD_75_135 --hemi lh --ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/colortable_BA.txt --l lh.BA1_exvivo.label --l lh.BA2_exvivo.label --l lh.BA3a_exvivo.label --l lh.BA3b_exvivo.label --l lh.BA4a_exvivo.label --l lh.BA4p_exvivo.label --l lh.BA6_exvivo.label --l lh.BA44_exvivo.label --l lh.BA45_exvivo.label --l lh.V1_exvivo.label --l lh.V2_exvivo.label --l lh.MT_exvivo.label --l lh.perirhinal_exvivo.label --l lh.entorhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s NIHPD_75_135 --hemi lh --ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/colortable_BA.txt --l lh.BA1_exvivo.thresh.label --l lh.BA2_exvivo.thresh.label --l lh.BA3a_exvivo.thresh.label --l lh.BA3b_exvivo.thresh.label --l lh.BA4a_exvivo.thresh.label --l lh.BA4p_exvivo.thresh.label --l lh.BA6_exvivo.thresh.label --l lh.BA44_exvivo.thresh.label --l lh.BA45_exvivo.thresh.label --l lh.V1_exvivo.thresh.label --l lh.V2_exvivo.thresh.label --l lh.MT_exvivo.thresh.label --l lh.perirhinal_exvivo.thresh.label --l lh.entorhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.stats -b -a ./lh.BA_exvivo.annot -c ./BA_exvivo.ctab NIHPD_75_135 lh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.thresh.stats -b -a ./lh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab NIHPD_75_135 lh white 

#--------------------------------------------
#@# BA_exvivo Labels rh Mon May 11 12:18:12 EEST 2026

 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA1_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA2_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA3a_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA3a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA3b_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA3b_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA4a_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA4a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA4p_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA4p_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA6_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA6_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA44_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA44_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA45_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA45_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.V1_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.V1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.V2_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.V2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.MT_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.MT_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.entorhinal_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.entorhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.perirhinal_exvivo.label --trgsubject NIHPD_75_135 --trglabel ./rh.perirhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.FG1.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./rh.FG1.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.FG2.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./rh.FG2.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.FG3.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./rh.FG3.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.FG4.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./rh.FG4.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.hOc1.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./rh.hOc1.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.hOc2.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./rh.hOc2.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.hOc3v.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./rh.hOc3v.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.hOc4v.mpm.vpnl.label --trgsubject NIHPD_75_135 --trglabel ./rh.hOc4v.mpm.vpnl.label --hemi rh --regmethod surface 


 mris_label2annot --s NIHPD_75_135 --ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/colortable_vpnl.txt --hemi rh --a mpm.vpnl --maxstatwinner --noverbose --l rh.FG1.mpm.vpnl.label --l rh.FG2.mpm.vpnl.label --l rh.FG3.mpm.vpnl.label --l rh.FG4.mpm.vpnl.label --l rh.hOc1.mpm.vpnl.label --l rh.hOc2.mpm.vpnl.label --l rh.hOc3v.mpm.vpnl.label --l rh.hOc4v.mpm.vpnl.label 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA1_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA2_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA3a_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA3a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA3b_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA3b_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA4a_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA4a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA4p_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA4p_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA6_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA6_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA44_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA44_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.BA45_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.BA45_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.V1_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.V1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.V2_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.V2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.MT_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.MT_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.entorhinal_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.entorhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /home/neurosign/FS_SUBS_DIR/fs741/fsaverage/label/rh.perirhinal_exvivo.thresh.label --trgsubject NIHPD_75_135 --trglabel ./rh.perirhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mris_label2annot --s NIHPD_75_135 --hemi rh --ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/colortable_BA.txt --l rh.BA1_exvivo.label --l rh.BA2_exvivo.label --l rh.BA3a_exvivo.label --l rh.BA3b_exvivo.label --l rh.BA4a_exvivo.label --l rh.BA4p_exvivo.label --l rh.BA6_exvivo.label --l rh.BA44_exvivo.label --l rh.BA45_exvivo.label --l rh.V1_exvivo.label --l rh.V2_exvivo.label --l rh.MT_exvivo.label --l rh.perirhinal_exvivo.label --l rh.entorhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s NIHPD_75_135 --hemi rh --ctab /home/neurosign/Softwares/FreeSurf/fs741/freesurfer/average/colortable_BA.txt --l rh.BA1_exvivo.thresh.label --l rh.BA2_exvivo.thresh.label --l rh.BA3a_exvivo.thresh.label --l rh.BA3b_exvivo.thresh.label --l rh.BA4a_exvivo.thresh.label --l rh.BA4p_exvivo.thresh.label --l rh.BA6_exvivo.thresh.label --l rh.BA44_exvivo.thresh.label --l rh.BA45_exvivo.thresh.label --l rh.V1_exvivo.thresh.label --l rh.V2_exvivo.thresh.label --l rh.MT_exvivo.thresh.label --l rh.perirhinal_exvivo.thresh.label --l rh.entorhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.stats -b -a ./rh.BA_exvivo.annot -c ./BA_exvivo.ctab NIHPD_75_135 rh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.thresh.stats -b -a ./rh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab NIHPD_75_135 rh white 

