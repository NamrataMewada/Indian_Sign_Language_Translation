// Navbar scroll effect
window.addEventListener("scroll", () => {
  const navbar = document.getElementById("navbar");
  if (window.scrollY > 50) {
    navbar.classList.add("scrolled");
  } else {
    navbar.classList.remove("scrolled");
  }
});

// Hamburger menu
const hamburger = document.querySelector(".hamburger");
const navLinks = document.querySelector(".nav-links");

hamburger.addEventListener("click", () => {
  navLinks.classList.toggle("active");
  hamburger.classList.toggle("active");
});


// Services page functionality
function showTranslation(type) {
  const signToTextSection = document.getElementById("signToTextSection");
  const textToSignSection = document.getElementById("textToSignSection");
  const serviceCards = document.getElementById("serviceCards");
  const backButton = document.getElementById("backToServices");
  const serviceIntro = document.getElementById("service-intro");

  if (type === "sign-to-text") {
    signToTextSection.style.display = "block";
    textToSignSection.style.display = "none";
    serviceCards.style.display = "none";
    backButton.style.display = "block";
    serviceIntro.style.display = "none";
    stopCamera(); // Stop any existing camera feed
  } else if (type === "text-to-sign") {
    signToTextSection.style.display = "none";
    textToSignSection.style.display = "block";
    serviceCards.style.display = "none";
    backButton.style.display = "block";
    serviceIntro.style.display = "none";
    stopCamera(); // Stop any existing camera feed
  } else {
    signToTextSection.style.display = "none";
    textToSignSection.style.display = "none";
    serviceCards.style.display = "grid";
    backButton.style.display = "none";
    serviceIntro.style.display = "block";
    stopCamera(); // Ensure camera is stopped when going back
  }
}

// Camera functionality
let translationInterval = null;
let lastSentence = "";

function injectRealTimeBoxes() {
  const typingContainer = document.getElementById("translationText");

  typingContainer.innerHTML = "";

  // ---Create live gesture element---
  const gestureBox = document.createElement("div");
  gestureBox.id = "liveGesture";
  gestureBox.className = "translation-gesture";
  gestureBox.innerText = "Detecting..";

  //--- Current Sentence---
  const currentText = document.createElement("div");
  currentText.id = "currentSentenceText";
  currentText.className = "translation-line";
  currentText.innerText = "Typing...";

  // ---Final Sentence---
  const finalBox = document.createElement("div");
  finalBox.id = "finalText";
  finalBox.className = "translation-text final-output";
  finalBox.innerText = "Awaiting final sentence...";

  // --- Previous Sentences Section ---
  const historyContainer = document.createElement("div");
  historyContainer.id = "allSentencesContainer";
  historyContainer.className = "sentence-history";

  const historyTitle = document.createElement("h4");
  historyTitle.innerText = "Previous Sentences";
  historyContainer.appendChild(historyTitle);

  const historyList = document.createElement("ul");
  historyList.id = "allSentencesList";
  historyContainer.appendChild(historyList);

  // Append all inside the same translation-text container
  typingContainer.appendChild(gestureBox);
  typingContainer.appendChild(currentText);
  typingContainer.appendChild(finalBox);
  typingContainer.appendChild(historyContainer);
}

async function accessCamera() {
  console.log("Camera function entered");
  const isLoggedIn = checkUserLogin();
  if (!isLoggedIn) {
    Swal.fire({
      icon: "warning",
      title: "Please login to access the Camera!",
      position: "top",
      toast: false,
      background: "#fff",
      color: "#323232",
      confirmButtonText: "Login Now",
      allowOutsideClick: false,
      allowEscapeKey: false,
      customClass: {
        container: "swal-login-alert",
      },
    }).then(() => {
      openAuthModal(); // Open login modal after confirmation
    });

    return;
  }

  try {
    // Chcek whether the user has granted the camera permission or not
    const permissionStatus = await navigator.permissions.query({
      name: "camera",
    });

    if (permissionStatus.state === "denied") {
      throw new Error("Camera permission denied");
    }
    console.log("access camera try block entered");
    stopCamera();

    // Add the UI blocks dynamically
    injectRealTimeBoxes();

    // Hide the video element is not used
    document.getElementById("videoFeed").style.display = "none";

    const cameraFeed = document.getElementById("cameraFeed");
    cameraFeed.style.display = "block";
    cameraFeed.src = "/video_feed";

    // Update button states
    document.getElementById("startCamera").style.display = "none";
    document.getElementById("stopCamera").style.display = "inline-block";
    document.getElementById("startTranslation").disabled = false;

    translationInterval = setInterval(updateTranslation, 500);
  } catch (error) {
    console.error("Camera permission denied or error accessing camera:", error);
    Swal.fire({
      icon: "error",
      title: "Camera Access Denied",
      text: "Please allow camera access in your browser settings to use this feature.",
      confirmButtonText: "OK",
      position: "top",
    });
  }
}

function stopCamera() {
  console.log("Stop camera is called");
  document.getElementById("cameraFeed").style.display = "none";
  document.getElementById("cameraFeed").src = "";

  // stops translation updates and display the final sentence
  if (translationInterval) {
    clearInterval(translationInterval);
    translationInterval = null;
  }

  // Reset button states
  const startCameraBtn = document.getElementById("startCamera");
  const stopCameraBtn = document.getElementById("stopCamera");
  if (startCameraBtn && stopCameraBtn) {
    startCameraBtn.style.display = "inline-block";
    stopCameraBtn.style.display = "none";
  }
}

// function to fetch translation from flask
async function updateTranslation() {
  try {
    console.log(
      "Code entered into the try block of updateTranslation function"
    );

    const response = await fetch("/get_translation");
    const data = await response.json();

    const gesture = data.gesture || "Detecting..";
    const currentText = data.current_text || "Typing...";
    const finalText = data.final_text || "Awaiting final sentence..";

    if (document.getElementById("liveGesture"))
      document.getElementById("liveGesture").innerText = gesture;

    if (document.getElementById("currentSentenceText"))
      document.getElementById("currentSentenceText").innerText = currentText;

    if (document.getElementById("finalText")) {
      // Update only if there is a new final sentence
      if (
        finalText !== "Awaiting final sentence.." &&
        finalText !== lastSentence
      ) {
        //============ NEW LOGIC ============
        const finalTextDiv = document.getElementById("finalText");
        finalTextDiv.innerHTML = ""; // Clear previous sentence

        // Split sentence into words and wrap each in a span
        finalText.split(" ").forEach((word) => {
          const span = document.createElement("span");
          span.className = "translation-item";
          span.innerText = word;
          finalTextDiv.appendChild(span);
          finalTextDiv.append(" "); // add space between words
        });

        lastSentence = finalText;
      }
    }

    // NEW: Fetch sentence history
    fetch("/get_all_sentences")
      .then((res) => res.json())
      .then((historyData) => {
        if (historyData.all_sentences) {
          const historyList = document.getElementById("allSentencesList");
          if (historyList) {
            historyList.innerHTML = "";
            historyData.all_sentences.forEach((sentence) => {
              const li = document.createElement("li");
              li.textContent = sentence;
              historyList.appendChild(li);
            });
          }
        }
      })
      .catch((err) => {
        console.log("Error in fetching all the sentences from the backend");
        console.error("Failed to fetch sentence history:", err);
      });
  } catch (error) {
    console.log("Error in catch block of update Translations");
    console.error("Error fetching the translation:", error);
  }
}

// Function to check if the user is logged in
function checkUserLogin() {
  return localStorage.getItem("token") !== null;
}

// Function to open login/signup modal
function openAuthModal() {
  document.getElementById("authModal").classList.add("show");
}

function uploadVideo() {
  document.getElementById("cameraFeed").style.display = "none";
  document.getElementById("videoFeed").style.display = "block";
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "video/*";

  const isLoggedIn = checkUserLogin();

  if (!isLoggedIn) {
    Swal.fire({
      icon: "warning",
      title: "Please login to upload the video!",
      position: "top",
      toast: false,
      background: "#fff",
      color: "#323232",
      confirmButtonText: "Login Now",
      allowOutsideClick: false,
      allowEscapeKey: false,
      customClass: {
        container: "swal-login-alert",
      },
    }).then(() => {
      openAuthModal(); // Open login modal after confirmation
    });
  } else {
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        const videoElement = document.getElementById("videoFeed");
        videoElement.src = URL.createObjectURL(file);
        document.getElementById("startTranslation").disabled = false;
        window.selectedVideo = file; // Store the selected video for upload
      }
    };

    input.click();
  }
}

let isTranslating = false;

async function startTranslation() {
  const translationText = document.getElementById("translationText");

  if (!isTranslating) {
    if (!window.selectedVideo) {
      translationText.innerText = "Please upload a video first.";
      return;
    }

    isTranslating = true;
    translationText.innerHTML =
      '<div class="translation-status">Translation in progress...</div>';

    const formData = new FormData();
    formData.append("video", window.selectedVideo);

    try {
      const response = await fetch("/process_video", {
        method: "POST",
        body: formData,
        headers: {
          Authoriation: `Bearer ${localStorage.getItem("token")}`,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to fetch translation.");
      }

      const data = await response.json();

      if (data.warning) {
        // showAlert(data.warning);
        Swal.fire({
          icon: "warning",
          title: "You have already uploaded the video",
          position: "top",
          toast: true,
          showConfirmButton: false,
          timer: 2500,
          timerProgressBar: true,
          background: "#fff",
          color: "#323232",
        });
      }
      translationText.innerHTML = `<div class="translation-item">${data.result}</div>`;
    } catch (error) {
      translationText.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }

    isTranslating = false;
  }
}

async function playTranslation() {
  const translationText = document.getElementById("translationText");
  const translations =
    translationText.getElementsByClassName("translation-item");

  if (translations.length === 0) {
    // alert("No translation available to play");
    Swal.fire({
      icon: "warning",
      title: "No translation available to play!",
      position: "top",
      toast: false,
      background: "#fff",
      color: "#323232",
      showConfirmButton: true,
    });
    return;
  }

  // Get the live translated sentence
  let sentence = Array.from(translations)
    .map((item) => item.innerText.trim())
    .join(" "); // Combine words into a sentence

  try {
    // Fetch the audio file from the backend
    const response = await fetch("/text-to-speech", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: sentence }),
    });

    const data = await response.json();
    if (data.error) {
      // alert("Error: " + data.error);
      Swal.fire({
        icon: "error",
        title: "Error: " + data.error,
        position: "top",
        showConfirmButton: true,
      });
      return;
    }

    const audio = new Audio(data.audio_url);
    audio.play();

    // Highlighting the logic for live sentence
    let currentIndex = 0;
    const playInterval = setInterval(() => {
      if (currentIndex < translations.length) {
        translations[currentIndex].classList.add("highlight");
        if (currentIndex > 0) {
          translations[currentIndex - 1].classList.remove("highlight");
        }
        currentIndex++;
      } else {
        clearInterval(playInterval);
        translations[translations.length - 1].classList.remove("highlight");
      }
    }, 1000);

    // === NEW LOGIC: Also play audio for previous sentences ===
    const allSentencesList = document.getElementById("allSentencesList");
    const finalTextDiv = document.getElementById("finalText");

    if (allSentencesList && allSentencesList.children.length === 0) {
      Swal.fire({
        icon: "warning",
        title: "No sentences available for playback!",
      });
      return;
    }

    // Clear current finalText display
    finalTextDiv.innerHTML = "";

    // Prepare the text
    let combinedText = "";
    Array.from(allSentencesList.children).forEach((li, index) => {
      const sentence = li.innerText.trim();
      if (!sentence) return;

      // Add sentence to combined text
      combinedText += sentence;
      if (index < allSentencesList.children.length - 1) {
        combinedText += ". "; // this add pause btween sentences
      }

      // Add each sentence to finalText with animation
      const span = document.createElement("span");
      span.className = "translation-item";
      span.innerText = sentence;
      finalTextDiv.appendChild(span);
      finalTextDiv.append(" "); // add space between senetence span
    });

    try {
      const response = await fetch("/text-to-speech", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: combinedText }),
      });
      const data = await response.json();
      const audio = new Audio(data.audio_url);
      audio.play();
    } catch (err) {
      console.error("Error during text to speech", err);
      Swal.fire({
        icon: "error",
        title: "Error playing audio",
      });
    }
  } catch (error) {
    console.error("Error fetching audio:", error);
  }
}

function resetSignToTextSection() {
  console.log("Resetting Sign to Text Section...");

  // 1. Stop camera and clear interval
  stopCamera();

  // 2. Hide video if any
  const videoFeed = document.getElementById("videoFeed");
  if (videoFeed) {
    videoFeed.style.display = "none";
    videoFeed.src = "";
  }

  // 3. Reset selected video
  window.selectedVideo = null;

  // 4. Reset translation UI content
  const translationText = document.getElementById("translationText");
  if (translationText) {
    translationText.innerHTML = "Translation will appear here...";
  }

  // 5. Re-disable startTranslation button
  const startBtn = document.getElementById("startTranslation");
  if (startBtn) startBtn.disabled = true;

  console.log("Reset complete.");
}

async function translateText(inputType = "text", audioPath = null) {
  const isLoggedIn = checkUserLogin();

  if (!isLoggedIn) {
    // showAlert("Please login for translation");
    // setTimeout(() => {
    //   openAuthModal();
    // }, 2000);

    Swal.fire({
      icon: "warning",
      title: "Please login for Traslation!",
      position: "top",
      toast: false,
      background: "#fff",
      color: "#323232",
      confirmButtonText: "Login Now",
      allowOutsideClick: false,
      allowEscapeKey: false,
      customClass: {
        container: "swal-login-alert",
      },
    }).then(() => {
      openAuthModal(); // Open login modal after confirmation
    });
  } else {
    const textInput = document.getElementById("textInput");
    const animationContainer = document.getElementById("animationContainer");

    const text = textInput.value.trim();
    if (!text) {
      // alert("Please enter text to translate");
      Swal.fire({
        icon: "warning",
        title: "Please enter text to translate",
        position: "top",
        showConfirmButton: true,
      });
      return;
    }

    let requestData = { text: text, input_type: inputType };

    if (audioPath) {
      requestData.audio_path = audioPath;
    }

    animationContainer.innerHTML =
      '<div class="animation-status">Generating sign language animation...</div>';

    try {
      const response = await fetch("/process_text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch translation.");
      }

      const data = await response.json();
      animationContainer.innerHTML = ""; // Clear previous content

      // Handle warnings from the backend
      if (data.warning) {
        // showAlert(data.warning); // Show warning message to the user
        Swal.fire({
          icon: "warning",
          title: data.warning,
          position: "top",
          toast: true,
          showConfirmButton: false,
          timer: 2500,
          timerProgressBar: true,
          background: "#fff",
          color: "#323232",
        });
      }

      const wordRow = document.createElement("div");
      wordRow.classList.add("word-row");

      data.result.forEach((item) => {
        const wordContainer = document.createElement("div");
        wordContainer.classList.add("word-container");

        const signMediaContainer = document.createElement("div");
        signMediaContainer.classList.add("sign-media-container");

        if (item.type === "gif") {
          // Display emergency sign as GIF
          const gifElement = document.createElement("img");
          gifElement.src = item.path;
          gifElement.classList.add("sign-gif");
          signMediaContainer.appendChild(gifElement);
        } else if (item.type === "images") {
          // Display letters as images side by side
          item.paths.forEach((path) => {
            const letterImg = document.createElement("img");
            letterImg.src = path;
            letterImg.classList.add("letter-img");
            signMediaContainer.appendChild(letterImg);
          });
        }

        // Add label below the word container
        const label = document.createElement("div");
        label.classList.add("animation-text");
        label.innerText = item.label;

        wordContainer.appendChild(signMediaContainer);
        wordContainer.appendChild(label);
        wordRow.appendChild(wordContainer);
        animationContainer.appendChild(wordContainer);
      });
    } catch (error) {
      animationContainer.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
  }
}

function playAnimation() {
  const isLoggedIn = checkUserLogin();
  if (!isLoggedIn) {
    // showAlert("Please Login to Record Speech");
    // setTimeout(() => {
    //   openAuthModal();
    // }, 2000);

    Swal.fire({
      icon: "warning",
      title: "Please login to Record Speech!",
      position: "top",
      toast: false,
      background: "#fff",
      color: "#323232",
      confirmButtonText: "Login Now",
      allowOutsideClick: false,
      allowEscapeKey: false,
      customClass: {
        container: "swal-login-alert",
      },
    }).then(() => {
      openAuthModal(); // Open login modal after confirmation
    });

    return;
  }

  document.getElementById("textInput").value = "Listening...";

  fetch("/speech-to-text", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${localStorage.getItem("token")}`,
    },
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        // alert("Error: " + data.error);
        Swal.fire({
          icon: "error",
          title: "Error" + data.error,
          position: "top",
          showConfirmButton: true,
        });
        document.getElementById("textInput").value = "Try again!";
      } else {
        if (!data.text || data.text.toLowerCase() === "undefined") {
          document.getElementById("textInput").value =
            "Try again. Speech not recorded!";
        } else {
          document.getElementById("textInput").value = data.text;
          translateText("speech", data.audio_path); // Only call if text is valid
        }
      }
    })
    .catch((error) => console.error("Speech Recognition Error:", error));
}

function resetTextToSignSection() {
  const textInput = document.getElementById("textInput");
  const animationContainer = document.getElementById("animationContainer");

  textInput.value = "";
  animationContainer.innerHTML = "Animation will appear here...";
}

// Feedback form submission
function submitFeedback(event) {
  event.preventDefault();

  const isLoggedIn = checkUserLogin();
  if (!isLoggedIn) {
    // showAlert("Please Login to submit feedback !!");
    // setTimeout(() => {
    //   openAuthModal();
    // }, 2000);
    // return;

    Swal.fire({
      icon: "warning",
      title: "Please login to submit the feedback!",
      position: "top",
      toast: false,
      background: "#fff",
      color: "#323232",
      confirmButtonText: "Login Now",
      allowOutsideClick: false,
      allowEscapeKey: false,
      customClass: {
        container: "swal-login-alert",
      },
    }).then(() => {
      openAuthModal(); // Open login modal after confirmation
    });
  }

  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;
  const category = document.getElementById("category").value;
  const rating = document.querySelector('input[name="rating"]:checked')?.value;
  const message = document.getElementById("message").value;

  // Validate rating selection
  if (!rating) {
    // alert("Please select a rating.");
    Swal.fire({
      icon: "warning",
      title: "Please select a rating.",
      position: "top",
      toast: true,
      background: "#fff",
      color: "#323232",
      timer: 2000,
      timerProgressBar: false,
      showConfirmButton: false,
    });
    return;
  }

  const feedbackData = {
    name,
    email,
    category,
    rating,
    message,
  };

  fetch("/submit-feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(feedbackData),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        // alert("Error: " + data.error);
        Swal.fire({
          icon: "error",
          title: "Error: " + data.error,
          position: "top",
          toast: true,
          background: "#fff",
          color: "#323232",
          timer: 2000,
        });
      } else {
        // alert("Thank you for your feedback!");
        Swal.fire({
          icon: "success",
          title: "Thank you for your feedback!",
          position: "top",
          toast: true,
          background: "#fff",
          timer: 2000,
        });
        event.target.reset(); // Reset the form
      }
    })
    .catch((error) => {
      console.error("Error submitting feedback:", error);
      // alert("An error occurred while submitting your feedback.");
      Swal.fire({
        icon: "error",
        title: "An error occurred while submitting your feedback.",
        position: "top",
        toast: true,
        background: "#fff",
      });
    });
}

// Login Modal Script Auth Modal Handling
const authModal = document.getElementById("authModal");
const authBtn = document.querySelector(".login-btn");
const closeModal = document.querySelector(".close-modal");

const modalContent = document.getElementById("container");
const registerBtn = document.getElementById("register");
const loginBtn = document.getElementById("login");

registerBtn.addEventListener("click", () => {
  container.classList.add("active");
});

loginBtn.addEventListener("click", () => {
  container.classList.remove("active");
});

// Toggle modal
if (authBtn) {
  authBtn.addEventListener("click", () => {
    authModal.classList.add("show");
    document.querySelector('[data-tab="login"]').click();
  });
}

if (closeModal) {
  closeModal.addEventListener("click", () => {
    authModal.classList.remove("show");
  });
}

// Close modal when clicking outside
window.addEventListener("click", (e) => {
  if (e.target === authModal) {
    authModal.classList.remove("show");
  }
});

// Load user details
function loadProfileData() {
  // document.getElementById("profileUsernameInput").value =
  //   localStorage.getItem("username") || "Guest";
  // document.getElementById("profileEmail").value =
  //   localStorage.getItem("email") || "example@example.com";
  // document.getElementById("newPassword").value = ""; // Clear password field

  const username = document.getElementById("profileUsernameInput");
  const email = document.getElementById("profileEmail");
  const newPassword = document.getElementById("newPassword");

  username.value = localStorage.getItem("username") || "Unknown_User";
  email.value = localStorage.getItem("email") || "Not found email";
  newPassword.value = "";
}

document.addEventListener("DOMContentLoaded", function () {
  const authModal = document.getElementById("authModal");
  const signUpForm = document.querySelector(".sign-up form");
  const signInForm = document.querySelector(".sign-in form");
  const modalContent = document.getElementById("container");

  const profileMenu = document.getElementById("profileMenu");
  const profileUsername = document.getElementById("profileUsername");
  const authButton = document.getElementById("authButton");
  const logoutButton = document.getElementById("logoutButton");

  // Remember me checkbox logic :
  const emailInput = document.getElementById("signInEmail");
  const passwordInput = document.getElementById("signInPassword");

  const rememberCheckbox = document.getElementById("rememberMe");

  // Load saved email from localStorage
  if (
    localStorage.getItem("rememberEmail") &&
    localStorage.getItem("rememberPassword")
  ) {
    emailInput.value = localStorage.getItem("rememberEmail");
    passwordInput.value = localStorage.getItem("rememberPassword");
    rememberCheckbox.checked = true;
  }

  // Function to check if the user is logged in
  function checkLoginStatus() {
    const token = localStorage.getItem("token");
    const username = localStorage.getItem("username");

    if (token && username) {
      authButton.style.display = "none"; // Hide login button
      profileMenu.style.display = "flex"; // Show profile icon
      // profileUsername.innerText = `Username: ${username}`; // Set username
      profileUsername.innerText = username;
    } else {
      authButton.style.display = "block"; // Show login button
      profileMenu.style.display = "none"; // Hide profile icon
    }
  }

  logoutButton.addEventListener("click", function () {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    authButton.style.display = "block";
    profileMenu.style.display = "none";
  });

  function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
  }

  function showError(input, message) {
    const errorDiv = document.createElement("div");
    errorDiv.className = "error-message";
    errorDiv.style.color = "red";
    errorDiv.style.fontSize = "12px";
    errorDiv.textContent = message;

    if (
      input.nextElementSibling &&
      input.nextElementSibling.className === "error-message"
    ) {
      input.nextElementSibling.remove();
    }
    input.parentNode.appendChild(errorDiv);
  }

  function clearErrors(form) {
    const errorMessages = form.querySelectorAll(".error-message");
    errorMessages.forEach((error) => error.remove());
  }

  // Signup AJAX Request
  signUpForm.addEventListener("submit", function (e) {
    e.preventDefault();
    clearErrors(signUpForm);

    const name = signUpForm
      .querySelector("input[placeholder='Name']")
      .value.trim();
    const email = signUpForm
      .querySelector("input[placeholder='Email']")
      .value.trim();
    const password = signUpForm
      .querySelector("input[placeholder='Password']")
      .value.trim();

    if (name === "" || !validateEmail(email) || password.length < 6) {
      if (name === "")
        showError(
          signUpForm.querySelector("input[placeholder='Name']"),
          "Name is required"
        );
      if (!validateEmail(email))
        showError(
          signUpForm.querySelector("input[placeholder='Email']"),
          "Valid email required"
        );
      if (password.length < 6)
        showError(
          signUpForm.querySelector("input[placeholder='Password']"),
          "Password must be at least 6 characters"
        );
      return;
    }

    fetch("/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: name,
        email: email,
        password: password,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.message) {
          // alert(data.message);
          Swal.fire({
            icon: "success",
            title: "Account created successfully",
            timer: 2000,
            showConfirmButton: false,
            position: "top",
          });
          modalContent.classList.remove("active"); // Switch to Sign In panel
        } else {
          // alert("Signup failed: " + (data.error || "Unknown error"));
          Swal.fire({
            icon: "error",
            title: "Account creation failed",
            text: data.error || "Unknown error",
            timer: 2000,
            showConfirmButton: false,
          });
        }
      })
      .catch((error) => console.error("Error:", error));
  });

  // Login AJAX Request
  signInForm.addEventListener("submit", function (e) {
    e.preventDefault();
    clearErrors(signInForm);

    const email = signInForm
      .querySelector("input[placeholder='Email']")
      .value.trim();
    const password = signInForm
      .querySelector("input[placeholder='Password']")
      .value.trim();

    if (!validateEmail(email) || password.length < 6) {
      if (!validateEmail(email))
        showError(
          signInForm.querySelector("input[placeholder='Email']"),
          "Valid email required"
        );
      if (password.length < 6)
        showError(
          signInForm.querySelector("input[placeholder='Password']"),
          "Password must be at least 6 characters"
        );
      return;
    }

    if (rememberCheckbox.checked) {
      localStorage.setItem("rememberEmail", emailInput.value);
      localStorage.setItem("rememberPassword", passwordInput.value);
    } else {
      localStorage.removeItem("rememberEmail");
      localStorage.removeItem("rememberPassword");
    }

    fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email, password: password }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.token) {
          // alert("Login successful!");
          Swal.fire({
            icon: "success",
            title: "Login Successful!",
            text: `Welcome back, ${data.username}!`,
            position: "top",
            // toast: true,
            // timer: 2000,
            showConfirmButton: true,
            // background: "#fff",
            // color: "#333",
          });

          localStorage.setItem("token", data.token); // Store JWT token
          localStorage.setItem("username", data.username);
          localStorage.setItem("email", data.email);

          checkLoginStatus();
          authModal.classList.remove("show"); // Close modal on success
        } else {
          // alert("Login failed: " + (data.error || "Invalid credentials"));
          Swal.fire({
            icon: "error",
            title: "Login Failed",
            text: data.error || "Invalid credentials",
            position: "top",
            toast: false,
            // timer: 2500,
            // showConfirmButton: true,
            customClass: {
              container: "swal-login-error",
            },
          });
        }
      })
      .catch((error) => console.error("Error:", error));
  });

  const profileButton = document.getElementById("profileButton");
  const profileModal = document.getElementById("profileModal");
  const closeProfileModal = document.querySelector(".close-modal");
  const cancelButton = document.getElementById("cancelButton");
  const saveChanges = document.getElementById("saveChanges");

  // Open Profile Modal
  profileButton.addEventListener("click", function () {
    console.log("Profile button clicked!!!");
    loadProfileData();
    profileModal.style.display = "flex";
  });

  // Close Modal
  closeProfileModal?.addEventListener("click", function () {
    profileModal.style.display = "none";
  });

  cancelButton?.addEventListener("click", function () {
    profileModal.style.display = "none";
  });

  // Save Changes (Optional Password Update)
  saveChanges?.addEventListener("click", async function (event) {
    event.preventDefault();
    const newUsername = document
      .getElementById("profileUsernameInput")
      .value.trim();
    const newEmail = document.getElementById("profileEmail").value.trim();
    const newPassword = document.getElementById("newPassword").value.trim();

    // if (newPassword) {
    //   localStorage.setItem("password", newPassword);
    //   // alert("Password updated successfully!");
    //   Swal.fire({
    //     icon: "success",
    //     title: "Password Updated successfully!!",
    //     position: "top",
    //     showConfirmButton: false,
    //     timer: 2000
    //     });
    // } else {
    //   // alert("Profile updated! (Password unchanged)");
    //   Swal.fire({
    //     icon: "success",
    //     title: "Profile updated! (Password unchanged)",
    //     position: "top",
    //     showConfirmButton: false,
    //     timer: 2000,
    //     toast: true
    //   });
    // }

    try {
      const response = await fetch("/update_profile", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: newUsername,
          email: newEmail,
          password: newPassword,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        // alert("Profile updated successfully!");
        Swal.fire({
          icon: "success",
          title: result.message,
          position: "top",
          showConfirmButton: false,
          timer: 2000,
        });

        // Update the displayed username in frontend
        document.getElementById("profileUsername").innerText = newUsername;
        localStorage.setItem("username", newUsername);
        localStorage.setItem("email", newEmail);
        localStorage.setItem("password", newPassword);
      } else {
        // alert("Error updating profile");
        Swal.fire({
          icon: "error",
          title: result.error || "Update failed!!",
          position: "top",
          showConfirmButton: true,
        });
      }
    } catch (err) {
      console.error(err);
      // alert("Error updating profile");
      Swal.fire({
        icon: "error",
        title: "Server Error",
        text: err.message,
      });
    }

    profileModal.style.display = "none";
  });

  // Close modal when clicking outside
  window.addEventListener("click", function (e) {
    if (e.target === profileModal) {
      profileModal.style.display = "none";
    }
  });
  checkLoginStatus();
});

document.querySelectorAll(".toggle-password").forEach((toggle) => {
  toggle.classList.add("fa-eye-slash"); // Set initial icon

  toggle.addEventListener("click", () => {
    const input = toggle.previousElementSibling;
    const isPassword = input.getAttribute("type") === "password";

    input.setAttribute("type", isPassword ? "text" : "password");

    // Toggle icons
    toggle.classList.toggle("fa-eye");
    toggle.classList.toggle("fa-eye-slash");
  });
});
