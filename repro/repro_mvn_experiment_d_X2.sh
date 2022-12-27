# Retrive device id
for i in "$@"
do
case $i in
    --device=*)
    DEVICE="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# Define configuration files path variables
CFG=config/runs/mvn_experiment.yaml


# Define output directories path variables
OUTDIR=experiments/data/outputs/d_X2/mvn_experiment


# Run experiments for multiple seeds
for SEED in {1..40};
do
  for dX2 in 1 2 3 4 5 6 7 8;
  do
    DIRNAME=seed_$SEED
    DIRNAME+=_d_X2_$dX2
    python run_mvn_experiment.py --cfg=$CFG --o=$OUTDIR/$DIRNAME --seed=$SEED --d_X2=$dX2
  done
done
