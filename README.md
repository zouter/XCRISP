X-CRISP: Domain-Adaptable and Interpretable
CRISPR Repair Outcome Prediction
==============================

X-CRISP is a neural network for predicting CRISPR repair outcomes based on target sequence, outcome, and microhomology (MH) characteristics. Outperforming prior models, X-CRISP highlights the importance of MH positions relative to the cut site over other characteristics when predicting repair outcomes. Through transfer learning, X-CRISP models can be adapted after pre-training on mouse embryonic stem cell data to other genomic contexts like human cell lines K562, HAP1, U2OS, and those with altered DNA repair function, showcasing improved performance when refined on as few as 50 target domain samples when compared to models with no refinement.
