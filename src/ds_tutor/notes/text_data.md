Text features are typically sparse and high-dimensional.
Raw strings should not be fed directly into most estimators.

TF-IDF (Term Frequency - Inverse Document Frequency) converts text
into weighted numeric vectors where common but uninformative words
receive less weight.

Use `handle_unknown='ignore'` style safeguards for categorical data,
and use text vectorizers for free-form language fields.

For mixed tabular datasets, isolate text columns into a dedicated
pipeline branch so numeric/categorical preprocessing stays stable.
