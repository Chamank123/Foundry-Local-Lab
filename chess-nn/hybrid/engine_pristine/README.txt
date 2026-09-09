Aryan's engine snapshot exactly as supplied in agent_1.zip, unmodified.

Byte-identical to git commit eb47019. Use THIS search_numba.py if the
hand-crafted engine is the submission: the NN-integrated copy in ../engine/
threads six weight arrays through every recursive search call, which is
behaviourally identical (same node counts at fixed depth) but 20-26% slower.
