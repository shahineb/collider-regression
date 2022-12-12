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
OUTDIR=experiments/data/outputs/mvn_experiment


# Run experiments for multiple seeds
for SEED in {1..100};
do
  DIRNAME=seed_$SEED
  python run_mvn_experiment.py --cfg=$CFG --o=$OUTDIR/$DIRNAME --seed=$SEED
done
