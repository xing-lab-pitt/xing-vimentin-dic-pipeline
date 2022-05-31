echo $1
for f in `ls $1`; do
    # may generate warning if file name does not have leading 0
    # this is fine, and dirs will not be moved anywhere
    mv $1$f $1$(echo $f | sed -e 's:^0*::')
done
