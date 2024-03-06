document.addEventListener("DOMContentLoaded", function() {
    // Get the dimensions of the image
    var moonboard = document.getElementById('moonboard');
    var moonboardWidth = moonboard.clientWidth;
    var moonboardHeight = moonboard.clientHeight;
  
  
    // Define the number of rows and columns
    var rows = 18;
    var cols = 11;
  
  
    // Calculate the dimensions of each grid square
    var squareWidth = moonboardWidth / cols - 5;
    var squareHeight = moonboardHeight / rows - 2.5;
  
  
    // Get the grid container
    var grid = document.getElementById('grid');
  
  
    // Array to store the coordinates of each square
    var tempCoordinates = [];
  
  
    // Array to store the clicked coordinates
    var coordinates = [];
  
  
    // Generate the grid squares dynamically
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var square = document.createElement('div');
        square.className = 'grid-square';
        square.style.width = squareWidth + 'px';
        square.style.height = squareHeight + 'px';
        square.style.top = ((i + 1) * squareHeight) + 'px';
        square.style.left = ((j + 1) * squareWidth) + 7 + 'px';
        grid.appendChild(square);
       
        // Calculate the coordinate of each square and push it to the coordinates array
        var letter = String.fromCharCode(65 + j);
        var number = rows - i;
        tempCoordinates.push(letter + number);
  
  
        // Add event listener to toggle the selection when the square is clicked
        square.addEventListener('click', function() {
          var index = Array.from(grid.children).indexOf(this);
          var square = this;
  
  
          // Toggle the selection
          if (square.classList.contains('clicked')) {
            square.classList.remove('clicked');
            // Remove coordinate from the clicked coordinates array
            var coordinateToRemove = tempCoordinates[index];
            coordinates = coordinates.filter(function(coordinate) {
              return coordinate !== coordinateToRemove;
            });
          } else {
            square.classList.add('clicked');
            // Add coordinate to the clicked coordinates array
            coordinates.push(tempCoordinates[index]);
          }
  
  
          // Update coordinate display
          document.getElementById('coordinateDisplay').textContent = 'Coordinates: ' + coordinates.join(', ');
        });
      }
    }
    // JavaScript: Add event listener to the "grade" button
    document.getElementById('gradeButton').addEventListener('click', function() {
      // Display "Finding beta..." while waiting for the response
      $('#grade').text('Finding beta...');
  
      // First AJAX request to find the beta
      $.ajax({
          url: '/find-beta', // Adjust this URL to your Flask route for finding beta
          type: 'POST',
          contentType: 'application/json',
          data: JSON.stringify({coordinates: coordinates}),
          success: function(betaResponse) {
              // Assuming betaResponse.beta contains the beta needed for grading
              // Display the beta response and proceed to grade the climb
              if (Array.isArray(betaResponse.grade)) {
                // Convert each element of the array to a string and join them with line breaks
                var betaStrings = betaResponse.grade.join(', ');
                $('#grade').html('<br><b>Beta found:</b> ' + betaStrings + '<br><br>Grading climb...');
              } else {
                // Fallback in case the response is not in the expected format
                $('#grade').text('Beta: ' + betaResponse.grade + '<br>Grading climb...');
              }
  
              // Now include the beta in the second AJAX request to grade the climb
              $.ajax({
                  url: '/grade-climb', // Ensure this matches your Flask route for grading
                  type: 'POST',
                  contentType: 'application/json',
                  // Include both coordinates and beta in the request data
                  data: JSON.stringify({
                      coordinates: coordinates
                  }),
                  success: function(gradeResponse) {
                      // Update HTML to display the grade
                      $('#grade').append('<br><b>Grade:</b> ' + gradeResponse.grade);
                  },
                  error: function(xhr, status, error) {
                      // Handle error or show a default error message for grading
                      console.error("AJAX request for grading failed:", error);
                      $('#grade').append('<br>Error grading climb');
                  }
              });
          },
          error: function(xhr, status, error) {
              // Handle error or show a default error message for finding beta
              console.error("AJAX request for finding beta failed:", error);
              $('#grade').text('Error finding beta');
          }
      });
  });
    document.getElementById('clearButton').addEventListener('click', function() {
      // Clear the coordinates array
      coordinates = [];

      $('#grade').text('');
  
      // Remove the 'clicked' class from all grid squares
      var allSquares = document.querySelectorAll('.grid-square.clicked');
      allSquares.forEach(function(square) {
          square.classList.remove('clicked');
      });
  
      // Update coordinate display to show no coordinates
      document.getElementById('coordinateDisplay').textContent = 'Coordinates: ';
  });
  
});
  