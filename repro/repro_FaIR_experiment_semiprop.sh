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
CFG=config/runs/FaIR_experiment.yaml


# Define output directories path variables
OUTDIR=experiments/data/outputs/semi_prop/FaIR_experiment


# Run experiments for multiple seeds
for SEED in {1..40};
do
  for SEMIPROP in 0 25 50 100 200 400 600 800 1000;
  do
    DIRNAME=seed_$SEED
    DIRNAME+=_semi_prop_$SEMIPROP
    python run_FaIR_experiment.py --cfg=$CFG --o=$OUTDIR/$DIRNAME --seed=$SEED --semi_prop=$SEMIPROP
  done
done
