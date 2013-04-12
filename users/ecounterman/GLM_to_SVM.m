function GLM_to_SVM(subject,session)

datadir = sprintf('/Volumes/Plata1/Metacontrast/Scans/%s_%s_Session/%s_%s_n', subject, session, subject, session);
cd(datadir)

nRuns = 10;
trialsPerRun = 56;
numConds = 14;


for i = 1:nRuns
    parfile{i} = sprintf('Stimuli/parfiles/%s_%s_singletrial_run%02d.par',subject,session,i);
    fid = fopen(parfile{i});
    stimOrder(:,:,i) = textscan(fid, '%f%f%s');
    fclose(fid);
        
    datafile{i} = sprintf('SVM_Analysis/leftV1Targ_meta%d_multiVoxFigData.mat', i);
    figData(i) = load(datafile{i});
end

numVoxels = size(figData(1).tSeries,2);
betasMatrix = zeros((nRuns*trialsPerRun*numConds/2),numVoxels,numConds/2);



end

