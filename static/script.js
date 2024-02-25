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
        // AJAX request to send coordinates to backend
        console.log(JSON.stringify({coordinates: coordinates}))
          $.ajax({
            url: '/grade-climb', // Ensure this matches the Flask route exactly
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ coordinates: coordinates }),
            success: function(response) {
                // Update HTML to display the grade
                $('#grade').text(response.grade);
            },
            error: function(xhr, status, error) {
                console.error("AJAX request failed:", error);
            }
          });
      
    });
});
  