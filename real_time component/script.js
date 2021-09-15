
  const videoElement = document.getElementById("video");
  const myImg = document.getElementById("my_img");
  const canvas = document.getElementById("cv1");
  const ctx = canvas.getContext("2d");
  const preprocessCheckbox = document.querySelector("input[name=preprocess]");
  const automodeCheckbox = document.querySelector("input[name=automode]");
  const showImageCheckbox = document.querySelector("input[name=showImage]");
	const showAreaCheckbox = document.querySelector("input[name=showArea]");
  let previousTime = 0;
  
  var displayMediaOptions = {
    video: {
      cursor: 'always'
    },
   audio: false
  }

  let videoLoaded = false;
canvas.addEventListener("click", function(e) {
  if (!videoLoaded) {
    startCapture();
  }
})


  async function startCapture(){
    try {
      videoElement.srcObject = await navigator.mediaDevices.getDisplayMedia(displayMediaOptions)
    }catch(err) {
      console.error("Error" + err)
    }
  }

  function videoOnLoad(element) {
    videoLoaded = true;
    resizeCanvas(element);
  }

  function resizeCanvas(element)
{
  canvas.width = element.offsetWidth;
  canvas.height = element.offsetHeight;
}

videoElement.addEventListener("timeupdate", function () {
  try {
      const currentTime = parseFloat(videoElement.currentTime);
      console.log('previous:', previousTime);
      console.log('current:', currentTime);
      if ((currentTime - previousTime) >= 0.1) {
          // stopCapture()
          console.log('yes');
          var cv3 = document.createElement('canvas');
          cv3.width = 720;
          cv3.height = 450;
          var ctx3 = cv3.getContext('2d');
          ctx3.drawImage(videoElement, 0, 0, videoElement.videoWidth, videoElement.videoHeight, 0, 0, cv3.width, cv3.height);
          
          imageData = cv3.toDataURL('image/jpg');
          // Save image into localStorage
          // try {
          //   localStorage.setItem(currentTime.toString(), imageData);
          // } catch (e) {
          //   console.log("Storage failed: " + e);
          // }
          if (showImageCheckbox.checked) {
            myImg.hidden = false;
            myImg.src = imageData; 
          } else {
            myImg.hidden = true;
          }
      }
      previousTime = currentTime;
  } catch (err) {
      console.log("video has deleted: " + err);
  }
});

var canvasx = canvas.offsetLeft
var canvasy = canvas.offsetTop
var last_mousex = last_mousey = 0;
var mousex = mousey = 0;
var mousedown = false;
var rect = {}
var autoMode = automodeCheckbox.checked;
var preprocess = preprocessCheckbox.checked;
var imageData;

preprocessCheckbox.addEventListener('change', function() {
  preprocess = this.checked;
});

showImageCheckbox.addEventListener('change', function() {
  if (this.checked) {
    myImg.hidden = false;
    myImg.src = imageData;
  } else {
    myImg.hidden = true;
  }
});


showAreaCheckbox.addEventListener('change', function() {
  if (this.checked) {
    ctx.clearRect(0,0,canvas.width,canvas.height); //clear canvas
    ctx.beginPath();
    ctx.rect(rect.x,rect.y, rect.width, rect.height);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.stroke();
  } else if (rect.width > 0) {
    ctx.clearRect(0,0,canvas.width,canvas.height); //clear canvas
  }
});

automodeCheckbox.addEventListener('change', function() {
  if (this.checked) {
    setInterval(showStuff(rect), 500);
  } else {
    clearInterval();
  }
});


canvas.addEventListener("mouseup", function (e) {
  mousedown = false;
  if (rect.width > 0) {
    if (automodeCheckbox.checked) {
      console.log('its checked');
        // clearInterval();
        showLoadingOCR();
        setInterval(showStuff(rect), 500)
    } else {
      showLoadingOCR();
      showStuff(rect);
			hint.innerHTML = "Click on stream to refresh."
    }
  }
}, false);

canvas.addEventListener("mousedown", function (e) {
  last_mousex = parseInt(e.clientX-canvasx);
	last_mousey = parseInt(e.clientY-canvasy);
  mousedown = true;
}, false);

canvas.addEventListener("mousemove", function (e) {
  mousex = parseInt(e.clientX-canvasx);
	mousey = parseInt(e.clientY-canvasy);
    if(mousedown) {
        ctx.clearRect(0,0,canvas.width,canvas.height); //clear canvas
        ctx.beginPath();
        var width = mousex-last_mousex;
        var height = mousey-last_mousey;
        ctx.rect(last_mousex,last_mousey,width,height);
        rect = {x: last_mousex, y: last_mousey, width, height}
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}, false);

function showLoadingOCR() {
  output.innerHTML = "Extracting text..."
}

function showStuff({width, height, x, y}) {
  aspectRatioY = videoElement.videoHeight / canvas.height;
  aspectRatioX = videoElement.videoWidth / canvas.width;

  const offsetY = 0.5 * aspectRatioY;

  var cv2 = document.createElement('canvas');
    cv2.width = width*aspectRatioX;
    cv2.height = height*aspectRatioY;
    var ctx2 = cv2.getContext('2d');
    ctx2.drawImage(videoElement, x*aspectRatioX, (y+offsetY)*aspectRatioY, width*aspectRatioX, (height-offsetY)*aspectRatioY, 0, 0, cv2.width, cv2.height);
    
    if (preprocess) {
      ctx2.putImageData(preprocessImage(cv2), 0, 0);
    }
    imageData = cv2.toDataURL('image/jpg');
    if (showImageCheckbox.checked) {
      myImg.hidden = false;
      myImg.src = imageData; 
    } else {
      myImg.hidden = true;
    }
    recognize_image(imageData);
}

let textNode = document.createTextNode('Here I am');

// Tesseract
  const { createWorker } = Tesseract;

  function recognize_image(img) {

(async () => {
  //const worker = createWorker();
	const worker = createWorker({ 
    langPath: "./tessdata/",
    gzip: false,  
  });
  await worker.load();
    await worker.loadLanguage('jpn');
    await worker.initialize('jpn');
    await worker.setParameters({preserve_interword_spaces: '1'})
    const {
      data: { text },
    } = await worker.recognize(img);
    console.log(text);
    // writeIntoHtml("Result: " + text);
    output.innerHTML = text;
    await worker.terminate();
})();
  }

  function writeIntoHtml(text) {
    let div = document.createElement('div');
  div.className = "alert";
  div.innerHTML = text;

  document.body.append(div);
  }

  function preprocessImage(canvas) {
    const processedImageData = canvas.getContext('2d').getImageData(0,0,canvas.width, canvas.height);
    blurARGB(processedImageData.data, canvas, radius=1);
    dilate(processedImageData.data, canvas);
    invertColors(processedImageData.data);
    thresholdFilter(processedImageData.data, level=0.4);
    return processedImageData;
  }

  // from https://github.com/processing/p5.js/blob/main/src/image/filters.js
  function thresholdFilter(pixels, level) {
    if (level === undefined) {
      level = 0.5;
    }
    const thresh = Math.floor(level * 255);
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      const gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      let val;
      if (gray >= thresh) {
        val = 255;
      } else {
        val = 0;
      }
      pixels[i] = pixels[i + 1] = pixels[i + 2] = val;
    }
  }
  // from https://css-tricks.com/manipulating-pixels-using-canvas/
  function invertColors(pixels) {
    for (var i = 0; i < pixels.length; i+= 4) {
      pixels[i] = pixels[i] ^ 255; // Invert Red
      pixels[i+1] = pixels[i+1] ^ 255; // Invert Green
      pixels[i+2] = pixels[i+2] ^ 255; // Invert Blue
    }
  }

  // internal kernel stuff for the gaussian blur filter
  let blurRadius;
  let blurKernelSize;
  let blurKernel;
  let blurMult;

  // from https://github.com/processing/p5.js/blob/main/src/image/filters.js
  function buildBlurKernel(r) {
  let radius = (r * 3.5) | 0;
  radius = radius < 1 ? 1 : radius < 248 ? radius : 248;

  if (blurRadius !== radius) {
    blurRadius = radius;
    blurKernelSize = (1 + blurRadius) << 1;
    blurKernel = new Int32Array(blurKernelSize);
    blurMult = new Array(blurKernelSize);
    for (let l = 0; l < blurKernelSize; l++) {
      blurMult[l] = new Int32Array(256);
    }

    let bk, bki;
    let bm, bmi;

    for (let i = 1, radiusi = radius - 1; i < radius; i++) {
      blurKernel[radius + i] = blurKernel[radiusi] = bki = radiusi * radiusi;
      bm = blurMult[radius + i];
      bmi = blurMult[radiusi--];
      for (let j = 0; j < 256; j++) {
        bm[j] = bmi[j] = bki * j;
      }
    }
    bk = blurKernel[radius] = radius * radius;
    bm = blurMult[radius];

    for (let k = 0; k < 256; k++) {
      bm[k] = bk * k;
    }
  }
}

// from https://github.com/processing/p5.js/blob/main/src/image/filters.js
function blurARGB(pixels, canvas, radius) {
  const width = canvas.width;
  const height = canvas.height;
  const numPackedPixels = width * height;
  const argb = new Int32Array(numPackedPixels);
  for (let j = 0; j < numPackedPixels; j++) {
    argb[j] = getARGB(pixels, j);
  }
  let sum, cr, cg, cb, ca;
  let read, ri, ym, ymi, bk0;
  const a2 = new Int32Array(numPackedPixels);
  const r2 = new Int32Array(numPackedPixels);
  const g2 = new Int32Array(numPackedPixels);
  const b2 = new Int32Array(numPackedPixels);
  let yi = 0;
  buildBlurKernel(radius);
  let x, y, i;
  let bm;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      cb = cg = cr = ca = sum = 0;
      read = x - blurRadius;
      if (read < 0) {
        bk0 = -read;
        read = 0;
      } else {
        if (read >= width) {
          break;
        }
        bk0 = 0;
      }
      for (i = bk0; i < blurKernelSize; i++) {
        if (read >= width) {
          break;
        }
        const c = argb[read + yi];
        bm = blurMult[i];
        ca += bm[(c & -16777216) >>> 24];
        cr += bm[(c & 16711680) >> 16];
        cg += bm[(c & 65280) >> 8];
        cb += bm[c & 255];
        sum += blurKernel[i];
        read++;
      }
      ri = yi + x;
      a2[ri] = ca / sum;
      r2[ri] = cr / sum;
      g2[ri] = cg / sum;
      b2[ri] = cb / sum;
    }
    yi += width;
  }
  yi = 0;
  ym = -blurRadius;
  ymi = ym * width;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      cb = cg = cr = ca = sum = 0;
      if (ym < 0) {
        bk0 = ri = -ym;
        read = x;
      } else {
        if (ym >= height) {
          break;
        }
        bk0 = 0;
        ri = ym;
        read = x + ymi;
      }
      for (i = bk0; i < blurKernelSize; i++) {
        if (ri >= height) {
          break;
        }
        bm = blurMult[i];
        ca += bm[a2[read]];
        cr += bm[r2[read]];
        cg += bm[g2[read]];
        cb += bm[b2[read]];
        sum += blurKernel[i];
        ri++;
        read += width;
      }
      argb[x + yi] =
        ((ca / sum) << 24) |
        ((cr / sum) << 16) |
        ((cg / sum) << 8) |
        (cb / sum);
    }
    yi += width;
    ymi += width;
    ym++;
  }
  setPixels(pixels, argb);
}

function getARGB (data, i) {
  const offset = i * 4;
  return (
    ((data[offset + 3] << 24) & 0xff000000) |
    ((data[offset] << 16) & 0x00ff0000) |
    ((data[offset + 1] << 8) & 0x0000ff00) |
    (data[offset + 2] & 0x000000ff)
  );
};

function setPixels (pixels, data) {
  let offset = 0;
  for (let i = 0, al = pixels.length; i < al; i++) {
    offset = i * 4;
    pixels[offset + 0] = (data[i] & 0x00ff0000) >>> 16;
    pixels[offset + 1] = (data[i] & 0x0000ff00) >>> 8;
    pixels[offset + 2] = data[i] & 0x000000ff;
    pixels[offset + 3] = (data[i] & 0xff000000) >>> 24;
  }
};

  // from https://github.com/processing/p5.js/blob/main/src/image/filters.js
 function dilate(pixels, canvas) {
  let currIdx = 0;
  const maxIdx = pixels.length ? pixels.length / 4 : 0;
  const out = new Int32Array(maxIdx);
  let currRowIdx, maxRowIdx, colOrig, colOut, currLum;

  let idxRight, idxLeft, idxUp, idxDown;
  let colRight, colLeft, colUp, colDown;
  let lumRight, lumLeft, lumUp, lumDown;

  while (currIdx < maxIdx) {
    currRowIdx = currIdx;
    maxRowIdx = currIdx + canvas.width;
    while (currIdx < maxRowIdx) {
      colOrig = colOut = getARGB(pixels, currIdx);
      idxLeft = currIdx - 1;
      idxRight = currIdx + 1;
      idxUp = currIdx - canvas.width;
      idxDown = currIdx + canvas.width;

      if (idxLeft < currRowIdx) {
        idxLeft = currIdx;
      }
      if (idxRight >= maxRowIdx) {
        idxRight = currIdx;
      }
      if (idxUp < 0) {
        idxUp = 0;
      }
      if (idxDown >= maxIdx) {
        idxDown = currIdx;
      }
      colUp = getARGB(pixels, idxUp);
      colLeft = getARGB(pixels, idxLeft);
      colDown = getARGB(pixels, idxDown);
      colRight = getARGB(pixels, idxRight);

      //compute luminance
      currLum =
        77 * ((colOrig >> 16) & 0xff) +
        151 * ((colOrig >> 8) & 0xff) +
        28 * (colOrig & 0xff);
      lumLeft =
        77 * ((colLeft >> 16) & 0xff) +
        151 * ((colLeft >> 8) & 0xff) +
        28 * (colLeft & 0xff);
      lumRight =
        77 * ((colRight >> 16) & 0xff) +
        151 * ((colRight >> 8) & 0xff) +
        28 * (colRight & 0xff);
      lumUp =
        77 * ((colUp >> 16) & 0xff) +
        151 * ((colUp >> 8) & 0xff) +
        28 * (colUp & 0xff);
      lumDown =
        77 * ((colDown >> 16) & 0xff) +
        151 * ((colDown >> 8) & 0xff) +
        28 * (colDown & 0xff);

      if (lumLeft > currLum) {
        colOut = colLeft;
        currLum = lumLeft;
      }
      if (lumRight > currLum) {
        colOut = colRight;
        currLum = lumRight;
      }
      if (lumUp > currLum) {
        colOut = colUp;
        currLum = lumUp;
      }
      if (lumDown > currLum) {
        colOut = colDown;
        currLum = lumDown;
      }
      out[currIdx++] = colOut;
    }
  }
  setPixels(pixels, out);
};