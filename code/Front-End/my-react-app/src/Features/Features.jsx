import React from 'react';
import './Features.css';
import features from './figures/features.png'

function Feature(feature) {
  return (
    <div>
      <p className="Subtitle"> <b> <u> {feature["Name"]} </u></b> </p>
      <p style={{color:"white"}}>
        <br/>
        <b>Uses:</b> {feature["Uses"]}
        <br/>
        {feature["Description"]}
      </p>
    </div>
  )
}

function Contents() {
  let FeaturesList = [
    {Name: "Embedding Similarity", Uses: "emb_sim_qd, emb_sim_ad", Description: "Cosine embedding similarity between q and d, and between a and d. This uses the spaCy embeddings dataset, and returns a value corresponding to the semantic similarity of the two input phrases."},
    {Name: "Part-Of-Speech Similarity", Uses: "pos_sim_ad", Description: "Jaccard similarity of the POS tags between a and d. This helps to analyse grammatical and syntactic consistency e.g. 'positive and negative' is similar to 'regular and irregular' and are suitable mutual distractors."}, 
    {Name: "Edit Distance", Uses: "edit_distance", Description: "Levenshtein distance between a and d. This is expected to be low for good distractors e.g. 'exothermic' and 'endothermic' are good mutual distractors and have an edit distance of only two."},
    {Name: "Token Similarity", Uses: "token_sim_qd, token_sim_ad", Description: "Jaccard similarity between q and d's tokens, and between a and d's tokens. This measures the rate of shared words between phrases e.g. 'central nervous system' and 'autonomic nervous system' are good mutual distractors and share two words."},
    {Name: "Length", Uses: "character_length_d, character_length_diff, token_length_d, token_length_diff", Description: "The length of d as a total character count, and as a total token count. Additionally, the difference in character/token count between a and d, with the motivation of expecting suitable distractors to be of similar lengths to their answer e.g. 'atoms' is a poor distractor for 'reflection, refraction and deflection'."},
    {Name: "Suffix Length", Uses: "abs_comm_suffix, rel_comm_suffix", Description: "The absolute and relative lengths of the longest common suffix between a and d, as a number of characters. This is motivated by a suitable distractor sharing common words or word endings with the correct answer e.g. 'stratosphere' and 'troposphere'."},
    {Name: "Word Frequency", Uses: "word_freq_d", Description: "Average frequency of all tokens in d, over a series of datasets including the support paragraphs provided in SciQ, and Wikipedia. This is a measure of how common a word is, which can approximate difficulty, with the motivation that words used frequently in texts are more widely applicable and so more often part of useful distractors."},
    {Name: "Plurality Consistency", Uses: "sing_plur", Description: "Consistent use of plural versus singular nouns between a and d. This has a value of 1 when both are singular, or both are plural, such that distractors are gramatically correct."},
    {Name: "Number Frequency", Uses: "number_d, number_diff", Description: "The appearance of numbers in d, and the consistent use of numbers between a and d. This is useful for determining values such as percentages, counts, and years - such that if an answer is one of these, it is highly likely that a suitable distractor will be."},
    {Name: "Wikipedia Embedding Similarity", Uses: "wikisim_entity, wikisim", Description: "Cosine embedding similarity between a and d using a Wikipedia dataset. If both a and d are Wikipedia entities (typically words with their own Wikipedia page), the similarity between these entities is calculated. Additionally, sentence embeddings are calculated in a similar manner to Embedding Similarity but with an orthogonal dataset."},
  ]

  let htmlFeature = new Array(FeaturesList.length)

  for (let i = 0; i < FeaturesList.length; i++) {
    htmlFeature[i] = Feature(FeaturesList[i])
  }

  return (
    <div className="Bio2">
      <div className="beforeTitle">
        <br/>
        <img className="title" src={features}/> 
      </div> 
      <div className="afterTitle">
        <br style={{content: "", margin: "2em", display: "block", "font-size": 5 }}/>
                {htmlFeature}
        <br/>
        .
      </div>
    </div> 
  )
}

function Features() {
  return (
    <div className="Bio">
      <Contents/>
    </div>
  );
}

export default Features;