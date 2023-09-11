load('trained_net.mat');
faceDetector = vision.CascadeObjectDetector();
videoReader = VideoReader('C:\Users\erdmg\Downloads\istockphoto-1388899176-640_adpp_is.mp4');

% Create a video player to display the processed frames
videoInfo = get(videoReader);
videoPlayer = vision.VideoPlayer('Position', [300 300 videoInfo.Width videoInfo.Height]);

while hasFrame(videoReader)
    frame = readFrame(videoReader);

    bbox = step(faceDetector, frame);

    for i = 1:size(bbox, 1)
        face = imcrop(frame, bbox(i, :));
        resizedFace = imresize(face, [224, 224]);

        % Perform your desired operations on the face region here

        frame = insertObjectAnnotation(frame, 'rectangle', bbox(i, :), '');
    end

    step(videoPlayer, frame);
end

release(videoReader);
release(videoPlayer);