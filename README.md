# Data Science Tutor

I've been learning data science for a while now. I've noticed that in different projects I might delve deep into a
particular step of the data science workflow, but the learnings are not carried over to the next project.

There's a misalignment between "2nd brain" tools like Obsidian and Notion and my needs. Having the notes and mathematics
in one place and executable code in another place is annoying.

This project is a kind of 2nd brain where I can gather robust data science practices, data visualisation aids and
mathematical reminders in one place.

## Goal

Create a python package that I can import into other projects that brings with it all the learning and creations I've
gathered and made. At its core is a base tutor class that acts as a kind of practical note on the subject. For example,
a MissingDataTutor class contains the code to create missing data visualisations, perhaps a reminder on which estimators
don't work with missing data and other bits of information that a tutor teaching about missing data would mention.

The tutors classes are then orchestrated by higher level experiment and EDA classes.

The package should help me to complete data science projects whilst reminding me of core mathematical concepts and
ensuring I'm not making wrong assumptions or falling for any gotchas.

It needs to operationalise my knowledge, enforce rigorous standards and act a flashcard for me. Not hiding the
implementation details.

## Code

## Outcomes

Project Context Class

- Creates train test split
  EDA Class
- Runs Mutual Information
- Flags missing values