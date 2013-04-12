function optseq2parfile(subjectID,scanDate,run)

optseqFile = uigetfile('*.*');
fid = fopen(['Stimuli/optseq/' optseqFile]);
stimOrder = textscan(fid, '%f%f%f%f%s');
fclose(fid);
% subjectID = input('Subject ID: ','s');
% scanDate = input('Scan date: ', 's');
% run = input('Run number: ');
nEvents = length(stimOrder{1});
trialNum = 1;

% write text file
fileName = sprintf('Stimuli/parfiles/%s_%s_singletrial_run%02d.par', subjectID, scanDate, run);
fid = fopen(fileName,'w');
for iEvent = 1:nEvents
    if strcmp(stimOrder{5}(iEvent),'NULL')
    else
        fprintf(fid, '%3.2f\t%d\t%s', stimOrder{1}(iEvent), trialNum+56*(run-1), char(stimOrder{5}(iEvent)));
%         fprintf(fid, '%3.2f\t%d\t%s', stimOrder{1}(iEvent)+358*(run-1), trialNum+56*(run-1), char(stimOrder{5}(iEvent)));
        fprintf(fid, '\n');
        trialNum = trialNum + 1;
    end
end
status = fclose(fid);

% report
if status==0
    fprintf('Wrote par file %s.\n', fileName)
else
    fprintf('Check par file %s.\n', fileName)
end

end