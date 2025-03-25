// Global variables to store options fetched from the API
let professionOptions = [];
let cityOptions = [];

// Fetch options from the backend and populate dropdowns
function fetchOptions() {
  fetch('/get_options')
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        console.error("Error fetching options:", data.error);
        return;
      }
      // Use only the top 30 values returned
      professionOptions = data.profession_options;
      cityOptions = data.city_options;
      populateDropdown('profession', professionOptions);
      populateDropdown('city', cityOptions);
    })
    .catch(error => console.error("Error fetching options:", error));
}

// Populate a dropdown element with options
function populateDropdown(elementId, optionsArray) {
  const selectElem = document.getElementById(elementId);
  selectElem.innerHTML = `<option value="">Select ${elementId.charAt(0).toUpperCase() + elementId.slice(1)}</option>`;
  optionsArray.forEach(option => {
    const opt = document.createElement('option');
    opt.value = option;
    opt.textContent = option;
    selectElem.appendChild(opt);
  });
}

// Mapping functions for simple categorical fields
function mapMarried(value) {
  return value.trim().toLowerCase() === "married" ? 1 : 0;
}

function mapHouse(value) {
  const val = value.trim().toLowerCase();
  if (val === "owned") return 2;
  if (val === "rented") return 1;
  if (val === "norent_noown") return 0;
  return 0;
}

function mapCar(value) {
  return value.trim().toLowerCase() === "yes" ? 1 : 0;
}

// One-hot encoding for dropdown-based categorical inputs
function oneHotEncode(inputValue, optionsArray) {
  const encoding = new Array(optionsArray.length).fill(0);
  const index = optionsArray.findIndex(item => item.toLowerCase() === inputValue.trim().toLowerCase());
  if (index !== -1) {
    encoding[index] = 1;
  }
  return encoding;
}

function makePrediction() {
  // Gather values from inputs
  const income = parseFloat(document.getElementById("income").value);
  const age = parseFloat(document.getElementById("age").value);
  const experience = parseFloat(document.getElementById("experience").value);
  const marriedInput = document.getElementById("married").value;
  const houseInput = document.getElementById("house_ownership").value;
  const carInput = document.getElementById("car_ownership").value;
  const state = parseFloat(document.getElementById("state").value);
  const currentJobYears = parseFloat(document.getElementById("current_job_years").value);
  const currentHouseYears = parseFloat(document.getElementById("current_house_years").value);

  const professionInput = document.getElementById("profession").value;
  const cityInput = document.getElementById("city").value;

  // Map the simple categorical fields
  const marriedMapped = mapMarried(marriedInput);
  const houseMapped = mapHouse(houseInput);
  const carMapped = mapCar(carInput);

  // One-hot encode the dropdown fields using the fetched options
  const professionEncoded = oneHotEncode(professionInput, professionOptions);
  const cityEncoded = oneHotEncode(cityInput, cityOptions);

  // Construct the final features array in the correct order:
  // [income, age, experience, married, house_ownership, car_ownership, state, current_job_years, current_house_years]
  // + (30 one-hot for profession) + (30 one-hot for city)
  const features = [
    income, age, experience,
    marriedMapped, houseMapped, carMapped, state,
    currentJobYears, currentHouseYears,
    ...professionEncoded,
    ...cityEncoded
  ];

  // Validate feature count
  if (features.length !== 69) {
    alert("Error: Expected 69 features but got " + features.length);
    return;
  }

  // Send the features to the backend
  fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ features: features })
  })
    .then(response => response.json())
    .then(data => {
      const resultDiv = document.getElementById("result");
      if (data.error) {
        resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
      } else {
        resultDiv.innerHTML = `<p>Status: ${data.message}</p>`;
      }
    })
    .catch(error => {
      console.error('Error:', error);
      document.getElementById("result").innerHTML = `<p style="color: red;">An error occurred.</p>`;
    });
}

// Fetch dropdown options when the page loads
window.onload = fetchOptions;
