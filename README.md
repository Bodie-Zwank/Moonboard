<h1>This project gives difficulty grades for climbs on the 2016 Moonboard Set.</h1>

<h4>The Moonboard is a standardized board for rock climbing, meaning all the holds, positions/orientations of holds, and angles are the same regardless of where you go.</h4>
<p>I wanted to use AI to help give more "objective" grades of difficulty for new climbs people may set. The first model was a neural network I created myself, which was a simple MLP model (in helpers/small). I later tried implementing a RNN, but fell victim to the vanishing gradient issue and saw no improvements. I later found that Github user jrchang612 had previously implemented "DeepRouteSet", a website that generated climbs for users (but didn't allow them to input their own). However, within his repository (available at https://github.com/jrchang612/MoonBoardRNN?tab=readme-ov-file), there was a more sophisticated model for grading climbs, which I cloned and implemented in my project (in helpers/big).</p> 
<h2>To use:</h2>
<p>Install dependencies from requirements.txt (ideally in a virtual environment). Run application.py in your terminal. This will start the Flask server available locally at http://127.0.0.1:5000</p>
