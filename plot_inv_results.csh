#!/bin/csh

gmtset HEADER_FONT_SIZE 16
gmtset LABEL_FONT_SIZE 12
gmtset ANNOT_FONT_SIZE 12
gmtset HEADER_OFFSET -0.2
gmtset BASEMAP_TYPE fancy

#makecpt -C/work3/wang/code_bkup/ToolKit/Models/ETOPO1/ETOPO1.cpt -T-4000/3091/50 > ETOPO1.cpt
set sta_lst1 = station_7D.lst
set sta_lst2 = station_JdF_land.lst
set cmap = ETOPO1.cpt
set REG = -R228/238/39/50
set Pbound = /work3/wang/code_bkup/ToolKit/Models/Plates/PB2002_boundaries.gmt

set title='Station Map'

set output_ps_file = sta_map_amphi.ps
if (-f $output_ps_file) rm $output_ps_file

set lon1=`echo $REG | cut -d/ -f1 | sed s/'\-R'/''/` 
set lon2=`echo $REG | cut -d/ -f2`
set lat1=`echo $REG | cut -d/ -f3`
set lat2=`echo $REG | cut -d/ -f4`
set lon=`echo $lon1 $lon2 | awk '{print ($1+$2)/2}'`
set lat=`echo $lat1 $lat2 | awk '{print ($1+$2)/2}'`
set lats1 = 42.5
set lats2 = 47
set SCA = -JL$lon/$lat/${lats1}/${lats2}/4i

grdimage topo2.grd -C$cmap -Itopo_gradient.grd -X6 -Y9.0 $REG $SCA -K -P > $output_ps_file
psbasemap $REG $SCA -Ba2/a2:."$title":WeSn -V -O -K >> $output_ps_file

pscoast $SCA $REG -A100 -N1/3/0/0/0 -N2/3/0/0/0 -O -K -W3 >> $output_ps_file
psxy $Pbound $SCA $REG -W5/255/165/0 -M"99999 99999"  -O -K >> $output_ps_file

set clr = {black,red,forestgreen,steelblue,gray19}
psxy $sta_lst1 $SCA $REG -Sc.2 -W1 -G$clr[1] -O -K >> $output_ps_file
psxy $sta_lst2 $SCA $REG -St.2 -W1 -G$clr[2] -O -K >> $output_ps_file
gmtset ANNOT_FONT_SIZE 10
gmtset D_FORMAT %g
psscale  -B1000:"Topography (m)": -C$cmap -D5.1/-2./10.2/0.3h -O -K >> $output_ps_file
echo $output_ps_file
