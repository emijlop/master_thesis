# Category crosswalk

Maps each benchmark's native categories to a shared construct list (`category_harmonized`). **The crosswalk is an operationalization decision.** Every non-trivial row needs a justification, because it directly shapes the RQ2 results. The machine-readable version is `CROSSWALK` in `src/benchbias/schema.py`; keep the two in sync.

## Shared constructs

`age`, `disability`, `gender`, `sexual_orientation`, `nationality`, `physical_appearance`, `race_ethnicity`, `religion`, `ses`. Intersections are written `a×b`.

Constructs that exist only in one benchmark (e.g. geographic origin or profession in HESEIA) get **their own harmonized label rather than being forced into the nearest BBQ category**. Forcing them in would hide exactly the divergence RQ3 is looking for.

## BBQ

| Native | Harmonized | Note |
|---|---|---|
| Age | age | |
| Disability_status | disability | |
| Gender_identity | gender | includes some trans/nonbinary items; check whether to split |
| Sexual_orientation | sexual_orientation | |
| Nationality | nationality | |
| Physical_appearance | physical_appearance | |
| Race_ethnicity | race_ethnicity | |
| Religion | religion | |
| SES | ses | operationalized largely through occupations |
| Race_x_gender | race_ethnicity×gender | |
| Race_x_SES | race_ethnicity×ses | |

## HESEIA

_To fill after inspecting the label set._ Open questions:
- Geographic origin (*origen geográfico*): nationality, race_ethnicity, or its own label?
- Profession: its own label, or ses?
- Multi-label items: one row per label, or one row with the intersection label?

## WinoBias

| Native | Harmonized | Note |
|---|---|---|
| gender (pro/anti-stereotypical × occupation) | gender | binary; occupation acts as the stereotype carrier |
