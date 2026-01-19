 source .srs_2/bin/activate

python run_causal_discovery.py --csv Dataset/mesa_SRS_master.csv --outcome epslpscl5c --algo all --ci_test fisherz --alpha 0.01 --fci_mode possible


=======

python run_causal_discovery.py   --csv Dataset/mesa_SRS_master.csv   --outcome epslpscl5c   --algo all   --ci_test fisherz   --alpha 0.05   --max_depth 2   --ges_timeout_sec 60   --fci_mode possible

=====

1

python run_causal_discovery.py \
  --csv Dataset/mesa_SRS_master.csv \
  --outcome epslpscl5c \
  --algo all \
  --ci_test fisherz \
  --alpha 0.05 \
  --max_depth 2 \
  --ges_timeout_sec 60 \
  --fci_mode possible \
  --export_json outputs/discovery_epss.json


==========

2

python validate_with_llm.py --in outputs/discovery_epss.json --out outputs/validation_epss.json --use_llm 1

python validate_with_llm.py --in outputs/discovery_epss.json --out outputs/validation_epss.json --use_llm 1

=============

run causal discovery for sleepy5


python run_causal_discovery.py \
  --csv Dataset/mesa_SRS_master.csv \
  --outcome sleepy5 \
  --algo all \
  --ci_test fisherz \
  --alpha 0.05 \
  --max_depth 2 \
  --ges_timeout_sec 60 \
  --fci_mode possible \
  --export_json outputs/discovery_sleepy5.json


=======
Validate the new results (same pipeline)


python validate_with_llm.py \
  --in outputs/discovery_sleepy5.json \
  --out outputs/validation_sleepy5.json \
  --use_llm 1


==============


Run causal discovery for tired5



python run_causal_discovery.py \
  --csv Dataset/mesa_SRS_master.csv \
  --outcome tired5 \
  --algo notears \
  --lambda1 0.02 \
  --thr 0.01 \
  --topk 20 \
  --export_json outputs/discovery_tired5.json



Step 3️⃣ Validate causes (same pipeline)

python validate_with_llm.py \
  --in outputs/discovery_tired5.json \
  --out outputs/validation_tired5.json \
  --use_llm 1
---------------------------------------------------------------------------------


Run causal discovery for pqptrbsa

python run_causal_discovery.py \
  --csv Dataset/mros_master_SRS_ready_v1.csv \
  --dataset mros \
  --outcome pqptrbsa \
  --lambda1 0.02 \
  --thr 0.01 \
  --topk 20 \
  --export_json outputs/mros_discovery_pqptrbsa.json




python validate_with_llm.py \
  --in outputs/mros_discovery_pqptrbsa.json \
  --out outputs/mros_validation_pqptrbsa.json \
  --use_llm 1

--------------------------------------------------------------------------

Run causal discovery for Outcome 2 (pqpsqual)

python run_causal_discovery.py \
  --csv Dataset/mros_master_SRS_ready_v1.csv \
  --dataset mros \
  --outcome pqpsqual \
  --lambda1 0.02 \
  --thr 0.01 \
  --topk 20 \
  --export_json outputs/mros_discovery_pqpsqual.json




python validate_with_llm.py \
  --in outputs/mros_discovery_pqpsqual.json \
  --out outputs/mros_validation_pqpsqual.json \
  --use_llm 1


----------------------------------------------------------

pqpeffic


python run_causal_discovery.py \
  --csv Dataset/mros_master_SRS_ready_v1.csv \
  --dataset mros \
  --outcome pqpeffic \
  --lambda1 0.02 \
  --thr 0.01 \
  --topk 20 \
  --export_json outputs/mros_discovery_pqpeffic.json


python validate_with_llm.py \
  --in outputs/mros_discovery_pqpeffic.json \
  --out outputs/mros_validation_pqpeffic.json \
  --use_llm 1
============


