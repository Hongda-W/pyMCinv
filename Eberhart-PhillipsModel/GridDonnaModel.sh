#/bin/bash

# take in Donna Eberhart-Phillips's 3D velocity model and output the velocity profiles

infname="vlnzw2p2dnxyzltln.tbl.txt"
locxy="locs.xy"
awk 'NR>2 {print $(NF-2)}' $infname | uniq > depths.txt
for dep in `cat depths.txt`;
do
    # lon lat depth vp vs rho
    awk -v depth="${dep}" 'NR>2 {if ($(NF-2)==depth) print $NF,$(NF-1),$(NF-2),$1,$3,$4}' $infname > Velmodel_${dep}.xyz
    awk '{print $1,$2,$4}' Velmodel_${dep}.xyz | xyz2grd -Gtmpvp.nc -R176/180/-41/-37 -I0.2
    awk '{print $1,$2,$5}' Velmodel_${dep}.xyz | xyz2grd -Gtmpvs.nc -R176/180/-41/-37 -I0.2
    awk '{print $1,$2,$6}' Velmodel_${dep}.xyz | xyz2grd -Gtmprho.nc -R176/180/-41/-37 -I0.2
    grdtrack $locxy -Gtmpvp.nc -R176/180/-41/-37 -Z > locsvp.xyz
    grdtrack $locxy -Gtmpvs.nc -R176/180/-41/-37 -Z > locsvs.xyz
    grdtrack $locxy -Gtmprho.nc -R176/180/-41/-37 -Z > locsrho.xyz
    paste locs.xy locsvp.xyz locsvs.xyz locsrho.xyz | awk -v depth="${dep}" '{print $1,$2,$3,$4,$5,depth}' >> locprofiles.txt
    rm tmpvp.nc tmpvs.nc tmprho.nc locsvp.xyz locsvs.xyz locsrho.xyz
done

cat ${locxy} | while read loc
do
    outfname=`echo ${loc} | awk '{print "Velprofile_"$1"_"$2".txt"}'`
    grep "${loc}" locprofiles.txt | sort -n -k6 > $outfname
done
