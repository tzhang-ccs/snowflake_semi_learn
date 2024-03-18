for i in `seq 10 20`
do
    echo DO $i RUN
    python snowflake_semi.py -p train -s $i

done
