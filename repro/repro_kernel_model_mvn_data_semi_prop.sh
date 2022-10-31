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
CFG=config/kernel_model_mvn_data.yaml


# Define output directories path variables
OUTDIR=experiments/data/outputs/kernel_model_mvn_data/semiprop


# Run experiments for multiple seeds
for SEED in {1..20};
do
  for SEMIPROP in 0 0.5 1 2 4 8 16 24;
  do
    DIRNAME=seed_$SEED
    DIRNAME+=_semiprop_$SEMIPROP
    python run_kernel_model_mvn_data.py --cfg=$CFG --o=$OUTDIR/$DIRNAME --seed=$SEED --semi_prop=$SEMIPROP
  done
done
