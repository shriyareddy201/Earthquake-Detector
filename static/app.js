var map;
var markers = [];

function initMap() {
    map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: 20, lng: 80 },
        zoom: 5
    });
}

document.getElementById('earthquakeForm').addEventListener('submit', function(event) {
    event.preventDefault();
    findNearestEarthquake();
});

function findNearestEarthquake() {
    var userLat = parseFloat(document.getElementById('latitude').value);
    var userLng = parseFloat(document.getElementById('longitude').value);

    // Remove previous markers
    markers.forEach(function(marker) {
        marker.setMap(null);
    });

    // Make an AJAX request to find the nearest earthquake location
    $.post('/find_nearest', { latitude: userLat, longitude: userLng }, function(data) {
        var nearestLat = data.Latitude;
        var nearestLng = data.Longitude;
        var nearestLocation = data.Location;

        var marker = new google.maps.Marker({
            position: { lat: nearestLat, lng: nearestLng },
            map: map,
            title: 'Nearest Earthquake: ' + nearestLocation
        });

        markers.push(marker);
        alert('Nearest Earthquake: ' + nearestLocation);
    });
}
