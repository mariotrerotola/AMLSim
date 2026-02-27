package amlsim;

import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * AML Transaction repository (set of transactions) for performance optimizations
 */
public class TransactionRepository {

    public final int size;
    private int index = 0;
//    private DecimalFormat amt_fmt;

    private int count = 0;
    private int limit = Integer.MAX_VALUE;  // Number of transactions as buffer

    private long[] steps;
    private String[] descriptions;
    private double[] amounts;
    private String[] origIDs;
    private String[] destIDs;

    private float[] origBefore;
    private float[] origAfter;
    private float[] destBefore;
    private float[] destAfter;
    private boolean[] isSAR;
    private long[] alertIDs;

    private Long2IntOpenHashMap txCounter;
    private Long2IntOpenHashMap sarTxCounter;
    private BufferedWriter logWriter;

    TransactionRepository(int size) {
        this.txCounter = new Long2IntOpenHashMap();
        this.sarTxCounter = new Long2IntOpenHashMap();

        this.size = size;
        this.steps = new long[size];
        this.descriptions = new String[size];
        this.amounts = new double[size];
        this.origIDs = new String[size];
        this.destIDs = new String[size];

        this.origBefore = new float[size];
        this.origAfter = new float[size];
        this.destBefore = new float[size];
        this.destAfter = new float[size];
        this.isSAR = new boolean[size];
        this.alertIDs = new long[size];
    }

    void setLimit(int limit){
        this.limit = limit;
    }

    void initLogWriter(String logFileName) throws IOException {
        closeLogWriter();
        this.logWriter = new BufferedWriter(new FileWriter(logFileName));
        this.logWriter.write("step,type,amount,nameOrig,oldbalanceOrig,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isSAR,alertID\n");
        this.logWriter.flush();
    }

    void closeLogWriter() throws IOException {
        if (this.logWriter != null) {
            this.logWriter.close();
            this.logWriter = null;
        }
    }

    void addTransaction(long step, String desc, double amt, String origID, String destID, float origBefore,
                        float origAfter, float destBefore, float destAfter, boolean isSAR, long aid){
        if(count >= limit){
            if(count == limit){
                System.err.println("Warning: the number of output transactions has reached the limit: " + limit);
                flushLog();
                count++;
            }
            return;
        }

        this.steps[index] = step;
        this.descriptions[index] = desc;
        this.amounts[index] = amt;
        this.origIDs[index] = origID;
        this.destIDs[index] = destID;
        this.origBefore[index] = origBefore;
        this.origAfter[index] = origAfter;
        this.destBefore[index] = destBefore;
        this.destAfter[index] = destAfter;
        this.isSAR[index] = isSAR;
        this.alertIDs[index] = aid;

        if(isSAR){
            sarTxCounter.addTo(step, 1);
        }else if(!desc.contains("CASH-")) {
            txCounter.addTo(step, 1);  // Exclude cash transactions for counter
        }

        count++;
        index++;
        if(index >= size){
            flushLog();
        }
    }

    private double getDoublePrecision(double d) {
        // Round down amount to two digits (e.g. 12.3456 --> 12.34)
        // DecimalFormat will not be used because of its computation cost
        return (int)(d * 100) / 100.0;
    }

    void writeCounterLog(long steps, String logFile){
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(logFile));
            writer.write("step,normal,SAR\n");
            for(long i=0; i<steps; i++){
                int numTx = txCounter.get(i);
                int numSARTx = sarTxCounter.get(i);
                writer.write(i + "," + numTx + "," + numSARTx + "\n");
            }
            writer.flush();
        }catch(IOException e){
            e.printStackTrace();
        }
    }

    void flushLog(){
        if(this.index == 0){
            return;
        }
        if(this.logWriter == null){
            throw new IllegalStateException("Transaction log writer is not initialized");
        }
        // Flush transaction logs to the CSV file
        try {
            for(int i = 0; i < this.index; i++){
                logWriter.write(steps[i] + "," + descriptions[i] + "," + getDoublePrecision(amounts[i]) + "," +
                        origIDs[i] + "," + getDoublePrecision(origBefore[i]) + "," + getDoublePrecision(origAfter[i]) + "," +
                        destIDs[i] + "," + getDoublePrecision(destBefore[i]) + "," + getDoublePrecision(destAfter[i]) + "," +
                        (isSAR[i] ? "1" : "0") + "," + alertIDs[i] + "\n");
            }
            logWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        index = 0;
    }

}
