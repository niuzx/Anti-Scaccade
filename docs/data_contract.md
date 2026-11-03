# Private data contract

Supply a JSON array, one object per subject, in a fixed order:

- `subject_id`: unique pseudonymous string; no names, phone numbers or dates of birth.
- `folder`: directory relative to `--data-root`. Absolute paths and directory traversal are rejected.
- `y`: seven integers or nulls, in order visuospatial/executive, naming, attention, language, abstraction, delayed recall, orientation. Maxima are 5,3,6,3,2,5,6.
- `total`: the recorded integer MoCA total, 0 through 30. Do not replace it with the domain sum.
- `odor`: sixteen binary identification values.

Each subject folder must contain `trial1_eye.npy` through `trial40_eye.npy` and matching `trial1_angle.npy` through `trial40_angle.npy`. Arrays must be finite two-dimensional numeric arrays, respectively 12 and 8 channels. Paired trial lengths must agree.

The initializer validates all arrays and emits local audit metadata under an ignored audit directory. It reports missing labels and total-score discrepancies but never changes them. The source-sheet audit utility is included for provenance; it expects the original restricted table schema and is not required when using this pseudonymous contract.
