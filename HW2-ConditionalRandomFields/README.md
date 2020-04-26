# Assignment 2: Conditional random fields dialogue act tagging

See Assignment2-Description.pdf in this repo for a detailed description of this assignment. Below is an explanation of what each file does:

* baseline-tagger.py: a tagger that tagged each utterance using the following "baseline" set of features:
  * a feature for whether the speaker changed in comparison with the previous utterance
  * a feature marking the first utterance of the dialogue.
  * a feature for every original token in the utterance
    * essentially, a list of all tokens in the utterance--the utterance "Hello, how are you?" would have list ["Hello" "," "how" "are" "you" "?"] as a feature
  * a feature for every part of speech tag in the utterance 
    * a list of all part of speech tags in the utterance, wherein each part of speech tag corresponds to one token in the utterance--so for ["Hello" "," "how" "are" "you" "?"] there would be a corresponding list of the 6 part of speech tags represented by these tokens
* advanced-tagger.py: I explored potential features for boosting accuracy beyond that of the baseline-tagger, and implemented the set of advanced features that yielded the highest accuracy on the validation set. The set of features I settled on for my advanced-tagger included all features in the baseline-tagger, and additionally, the following:
  * a feature for whether the utterance contained a digit
  * a feature for whether the utterance contained a question mark
  * a feature for whether the utterance contained an exclamation mark
  * a feature for whether the utterance contained the token "uh-huh" (I learned that people say "uh-huh" a LOT in conversation!)
  * a feature for whether the utterance contained only a single word
 
Features I experimented with but ultimately rejected in the advanced-tagger included:
* replacing the feature for every original token in the utterance with a feature for every token in the utterance *converted to lowercase*
  * In the baseline-tagger, we were instructed to keep tokens in their original case--the utterance "Hello, how are you?" would be split into tokens ["Hello" "," "how" "are" "you" "?"]. This means the baseline-tagger would record "Hello" and "hello" as separate tokens, while in the advanced-tagger my experiment was to turn all "Hello" into "hello" and see if this improved accuracy.
* including both a feature for every original token in the utterance AND a feature for every token in the utterance *converted to lowercase*
  * This meant that utterance "Are you from the USA?" would have one feature with original tokens (["Are" "you" "from" "the" "USA" "?"]) and one feature with lowercased tokens (["are" "you" "from" "the" "usa" "?"]).
* similar to the "uh-huh" feature that made it into the advanced-tagger, I also tried checking whether the utterance contained a few other frequently-used single-word utterances, including "so" "yes" and "no." Interestingly, the "uh-huh" feature was the only feature of this nature that improved accuracy on the validation set.
