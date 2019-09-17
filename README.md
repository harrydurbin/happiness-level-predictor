<h2>World Happiness Survey Data</h2>
Data from the World Values Survey includes a collection of questionnaire results for over 86,000 people in 60 countries. The questionnaire asked hundreds of questions, one being, "how happy do you consider yourself currently?"
<br>
1 – very happy
<br>
2 – rather happy
<br>
3 – not very happy
<br>
4 – not at all happy
<br>

The data was used to:
<ol>
<li>Find features that correlate most with happiness.</li>
<li>Create model to predict subjective, self-reported happiness levels for individuals and averaged for countries.</li>
<li>Evaluate happiness levels in countries across the world to see spatial influence.</li>
</ol>
<img src="../master/img/happinessfactors.png?raw=true" width="100%"/>
<br>
This questionaire includes hundreds of other questions, yet some topics (e.g. health) are more closely linked to happiness.The dozen questions with the highest correlation to happiness were used to make a classification model to predict a person's happiness category (which can then be grouped for average country happiness). Various algorithms were investigated including random forest, decision tree, and nearest neighbor.
<br>

Links: 
<ul>
<li><a href="https://github.com/harrydurbin/happiness-level-predictor/blob/master/happiness.ipynb">iPython Notebook code</a></li>
<li><a href="https://happiness-predictor.herokuapp.com/">Web App to take survey and make happiness prediction</a></li>
<li><a href="http://arcg.is/SfmmL">ArcGIS Online Web Map</a></li>
</ul>



