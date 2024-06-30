const segment = document.getElementById("segment")
const overlay = document.getElementById("overlay")
const segErrorInfo = document.getElementById("segErrorInfo")
const errorInfo = document.getElementById("errorInfo")

const fileInput = document.getElementById("fileInput")
const uploadSection = document.getElementById("uploadSection")
const fileStatus = document.getElementById("fileStatus")
const spinner = document.getElementById("spinner")

const gridContainer = document.getElementById("gridContainer")
const origImage = document.getElementById("origImage")
const segment1 = document.getElementById("segmentation")
const recv1 = document.getElementById("recv1")
const recv2 = document.getElementById("recv2")

const inputContainer = document.getElementById("inputContainer")
const userInput = document.getElementById("userInput")
const recordButton = document.getElementById("recordButton")
const micIcon = document.getElementById("micIcon")
const sendArrow = document.getElementById("sendArrow")
const timer = document.getElementById("timer")
const ripple = document.getElementById("ripple")

const recordingTime = parseInt(recordButton.dataset.recordingTime, 10)

uploadSection.addEventListener("click", function () {
  fileInput.click()
})

let FILENAME = "test.jpg" // could be 'test.txt' or 'test.webm'
let mediaRecorder
let recording = false
let secondsElapsed = 0
let timerInterval
let audioChunks = []
let COUNTER = 0

// disable the text area and record button and enable after segment
userInput.disabled = true
recordButton.disabled = true

fileInput.addEventListener("change", checkFileSelected)
function checkFileSelected() {
  if (fileInput.files.length > 0) {
    const uploaded_file = fileInput.files[0]
    const fileType = fileInput.files[0].type
    const fileName = fileInput.files[0].name
    if (!fileType) {
      // Handle files without a reported MIME type
      segment.disabled = true
      fileStatus.classList.add("warning")
      fileStatus.innerText = "Unknown File Extension: " + fileName
      return
    }
    if (fileType.startsWith("image/")) {
      segment.disabled = false
      fileStatus.classList.remove("warning")
      fileStatus.innerText = ""

      const reader = new FileReader()
      reader.addEventListener("load", () => {
        origImage.src = reader.result
        segment1.style.opacity = 0.5
        recv1.style.opacity = 0.0
        recv2.style.opacity = 0.0
      })
      reader.readAsDataURL(uploaded_file)
    } else {
      segment.disabled = true
      fileStatus.classList.add("warning")
      fileStatus.innerText = "This File Type is not Supported: " + fileName
    }
  }
}

function updateElements() {
  // disable fileInput area
  uploadSection.style.cursor = "default"
  uploadSection.style.pointerEvents = "none"
  // enable text area and record button
  inputContainer.style.borderColor = "#28a745"
  inputContainer.style.backgroundColor = "#fff"
  userInput.disabled = false
  recordButton.disabled = false
  recordButton.style.cursor = "pointer"
}

segment.addEventListener("click", (e) => {
  fileInput.disabled = true
  segErrorInfo.classList.remove("active")
  spinner.style.display = "inline-block" // Show the spinner
  e.target.disabled = true

  FILENAME = Math.round(Math.random() * 10000) + fileInput.files[0].name

  // send the image
  const imgDataUrl = origImage.src
  // Convert Data URL to Blob
  fetch(imgDataUrl)
    .then((response) => response.blob()) // Convert to Blob
    .then(async (blob) => {
      const formData = new FormData()
      formData.append("image", blob, FILENAME)
      // Send image to backend
      try {
        const response = await fetch("/process_image", {
          method: "POST",
          body: formData,
        })
        if (response.ok) {
          const blob_1 = await response.blob() // Get the image blob

          const imgUrl = URL.createObjectURL(blob_1)
          segment1.src = imgUrl
          spinner.style.display = "none"
          segment1.style.opacity = 1.0
          updateElements()
          COUNTER++ // increment counter
        } else {
          // Non-OK response, it's a JSON object
          const errorData = await response.json()
          segErrorInfo.textContent = errorData.error
          segErrorInfo.classList.add("active")
          spinner.style.display = "none"
          fileInput.disabled = false
        }
      } catch (error) {
        segErrorInfo.textContent = `Error: ${error}`
        segErrorInfo.classList.add("active")
        spinner.style.display
        fileInput.disabled = false
      }
    })
})

function showToast(message) {
  const toastMessage = document.getElementById("toastMessage")
  const toastText = document.getElementById("toastText")
  const toastCloseBtn = document.getElementById("toastCloseBtn")

  toastText.textContent = message
  toastMessage.classList.remove("toast-hidden")
  toastMessage.classList.add("toast-show")

  toastCloseBtn.onclick = () => {
    toastMessage.classList.remove("toast-show")
    toastMessage.classList.add("toast-hidden")
  }

  // Hide the toast after 10 seconds
  setTimeout(() => {
    toastMessage.classList.remove("toast-show")
    toastMessage.classList.add("toast-hidden")
  }, 10000)
}

userInput.addEventListener("input", () => {
  sendArrow.style.display = userInput.value.trim() ? "block" : "none"
  if (!userInput.value.trim()) {
    userInput.placeholder = "Replace the person with a batman..."
    clearInterval(timerInterval) // Stop timer if running
    secondsElapsed = 0
  }
})

userInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey && userInput.value.trim()) {
    event.preventDefault() // Prevent new line
    await sendDataToServer(
      userInput.value.trim(),
      "text",
      FILENAME.split(".")[0] + ".txt"
    )
    userInput.value = "" // clear the user input
  }
})

sendArrow.addEventListener("click", async () => {
  if (userInput.value.trim()) {
    await sendDataToServer(
      userInput.value.trim(),
      "text",
      FILENAME.split(".")[0] + ".txt"
    )
    userInput.value = ""
  }
})

async function sendDataToServer(data, type, filename) {
  const formData = new FormData()
  // Create Blob for text data
  if (type == "text") {
    data = new Blob([data], { type: "text/plain;charset=utf-8" })
  }
  formData.append("data", data, filename)
  formData.append("dataType", type)

  overlay.style.display = "block"

  try {
    const response = await fetch("/process_text_audio", {
      method: "POST",
      body: formData,
    })

    if (response.ok) {
      // Handle response for both audio and text
      const imgBlob = await response.blob() // Get the image blob

      if (COUNTER < 3) {
        const generatedImg = document.getElementById(`recv${COUNTER}`)
        generatedImg.src = URL.createObjectURL(imgBlob)
        generatedImg.style.opacity = 1.0
      } else {
        // for counter >= 3
        // Create new div element
        const gridItem = document.createElement("div")
        gridItem.classList.add("grid-item")

        // Create img element and set src
        const img = document.createElement("img")
        img.src = URL.createObjectURL(imgBlob)
        img.alt = `Image ${COUNTER}`
        gridItem.appendChild(img)

        // Create p element for caption
        const caption = document.createElement("p")
        caption.textContent = `Generated Image ${COUNTER}.`
        gridItem.appendChild(caption)

        // Append div to container
        gridContainer.appendChild(gridItem)
      }
      COUNTER++
      overlay.style.display = "none"
    } else {
      // Non-OK response, it's a JSON error
      const errorData = await response.json()
      showToast(errorData.error)
      overlay.style.display = "none"
    }
  } catch (error) {
    overlay.style.display = "none"
    showToast(error)
    // console.log("Error:", error)
  }
}

function stopRecording() {
  if (mediaRecorder && recording) {
    mediaRecorder.stop() // stop ther recorder first
    recording = false // update the recording status
  }
  micIcon.classList.remove("ripple")

  // sendArrow.style.display = "block"
  recordButton.title = `Record Audio upto ${recordingTime}sec`
  userInput.disabled = false
  userInput.placeholder = "Audio recording sent!"
  userInput.value = ""
  secondsElapsed = 0
}

function startTimer() {
  timerInterval = setInterval(() => {
    // createRipple()
    secondsElapsed++
    userInput.value = `Listening...Time elapsed: ${secondsElapsed} Sec`
    if (secondsElapsed >= recordingTime) {
      clearInterval(timerInterval) // Clear the timer on stop
      stopRecording()
    }
  }, 1000)
}

// const audioPlayback = document.createElement("audio") // Create audio element
// function playAudio(blob) {
//   audioPlayback.src = URL.createObjectURL(blob) // Set audio source
//   audioPlayback.controls = true // Add playback controls
//   document.body.appendChild(audioPlayback) // Add to DOM
//   audioPlayback.play() // Start playback
// }

// Audio recording logic using MediaRecorder API

navigator.mediaDevices
  .getUserMedia({ audio: true })
  .then((stream) => {
    mediaRecorder = new MediaRecorder(stream) //, { mimeType: "audio/webm;codecs=opus" })

    recordButton.addEventListener("click", () => {
      if (!recording) {
        // Start recording logic here
        recording = true
        audioChunks = [] // Reset chunks for new recording
        mediaRecorder.start()

        sendArrow.style.display = "none"
        recordButton.title = "Stop Recording"
        userInput.disabled = true
        userInput.value = "Listening..."

        micIcon.classList.add("ripple")
        startTimer()
      } else {
        stopRecording() // Call your existing stopRecording function
      }
    })

    mediaRecorder.addEventListener("dataavailable", async (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data)
      }
      // If recording has stopped and data is available, create the blob
      if (!recording && audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, {
          type: "audio/webm;codecs=opus",
        })
        // playAudio(audioBlob) // Play the recorded audio
        await sendDataToServer(
          audioBlob,
          "audio",
          FILENAME.split(".")[0] + ".webm"
        ) // Send the audio blob to server
        audioChunks = [] // clear the array for the next recording
      }
    })
  })
  .catch((error) => {
    showToast("Error accessing microphone.")
  })
