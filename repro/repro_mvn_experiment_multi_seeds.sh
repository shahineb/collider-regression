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
OUTDIR=experiments/data/outputs/n_train/mvn_experiment


# Run experiments for multiple seeds
for SEED in {1..20};
do
  for NTRAIN in 20 50 100 150 200 300 400 600;
  do
    DIRNAME=seed_$SEED
    DIRNAME+=_n_$NTRAIN
    python run_mvn_experiment.py --cfg=$CFG --o=$OUTDIR/$DIRNAME --seed=$SEED --n=$NTRAIN
  done
done
