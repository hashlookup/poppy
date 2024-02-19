use std::{cmp::max, fmt::Display, fs::File, io::BufRead, mem::size_of, sync::Arc, thread};

use anyhow::anyhow;
use clap::Parser;
use poppy::{bloom, BloomFilter};

// When built with musl, the allocator slows down a lot insertion
// so we have to use jemalloc to get back to the expected performance
#[cfg(all(target_env = "musl", target_pointer_width = "64"))]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

pub enum ByteSize {
    Bytes(usize),
    Kilo(f64),
    Mega(f64),
    Giga(f64),
}

const KILO: usize = 1 << 10;
const MEGA: usize = 1 << 20;
const GIGA: usize = 1 << 30;

impl Display for ByteSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(b) => write!(f, "{}B", b),
            Self::Kilo(k) => write!(f, "{:.1}KB", k),
            Self::Mega(m) => write!(f, "{:.1}MB", m),
            Self::Giga(g) => write!(f, "{:.1}GB", g),
        }
    }
}

impl ByteSize {
    fn from_bytes(b: usize) -> Self {
        if b < KILO {
            Self::Bytes(b)
        } else if b < MEGA {
            Self::Kilo(b as f64 / KILO as f64)
        } else if b < GIGA {
            Self::Mega(b as f64 / MEGA as f64)
        } else {
            Self::Giga(b as f64 / GIGA as f64)
        }
    }
}

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Parser)]
enum Command {
    /// Create a new bloom filter
    Create(Create),
    /// Insert data into an existing bloom filter
    Insert(Insert),
    /// Checks entries against an existing bloom filter
    Check(Check),
    /// Show information about an existing bloom filter
    Show(Show),
}

#[derive(Debug, Parser)]
struct Create {
    /// Capacity of the bloom filter
    #[clap(short, long, default_value = "10000")]
    capacity: u64,
    /// False positive rate
    #[clap(short, long, default_value = "0.01")]
    probability: f64,
    /// File to store the bloom filter
    file: String,
}

#[derive(Debug, Parser)]
struct Insert {
    /// Show progress information
    #[clap(short, long)]
    progress: bool,
    /// The number of jobs to use to insert into the bloom filter. The original
    /// filter is copied into the memory of each job so you can expect the memory
    /// of the whole process to be N times the size of the (uncompressed) bloom filter.
    #[clap(short, long, default_value = "2")]
    jobs: usize,
    /// File containing the bloom filter to update
    file: String,
    /// Input files containing one entry per line
    inputs: Vec<String>,
}

#[derive(Debug, Parser)]
struct Check {
    /// Numbers of parallel jobs to check bloom filter
    #[clap(short, long, default_value = "2")]
    jobs: usize,
    /// Only show entries not in the bloom filter
    #[clap(long)]
    verify: bool,
    /// File containing the bloom filter
    file: String,
    /// Input files containing one entry per line
    inputs: Vec<String>,
}

#[derive(Debug, Parser)]
struct Show {
    /// File containing the bloom filter
    file: String,
}

fn main() -> Result<(), anyhow::Error> {
    let sub = Args::parse();

    match sub.command {
        Command::Create(o) => {
            let mut f = File::create(o.file)?;
            let b = bloom!(o.capacity, o.probability);
            b.write(&mut f)?;
        }
        Command::Insert(o) => {
            let bloom_file = File::open(&o.file)?;
            let bf = Arc::new(std::sync::Mutex::new(BloomFilter::from_reader(bloom_file)?));

            let mut handles = vec![];
            let files = o.inputs.clone();

            let batches = files.chunks(max(files.len() / o.jobs, 1));

            for batch in batches {
                let shared = Arc::clone(&bf);
                let batch: Vec<String> = batch.to_vec();
                let mut copy = shared
                    .lock()
                    .map_err(|e| anyhow!("failed to lock mutex: {}", e))?
                    .clone();

                let h = thread::spawn(move || {
                    {
                        //println!("Processing batch of {} files", batch.len());
                        for input in batch {
                            if o.progress {
                                println!("processing file: {input}");
                            }
                            let in_file = std::fs::File::open(&input)?;

                            for line in std::io::BufReader::new(in_file).lines() {
                                copy.insert(line?);
                            }
                        }

                        let mut shared = shared
                            .lock()
                            .map_err(|e| anyhow!("failed to lock mutex: {}", e))?;
                        shared.union(&copy)?;

                        Ok::<(), anyhow::Error>(())
                    }
                });
                handles.push(h)
            }

            for h in handles {
                h.join().expect("failed to join thread")?;
            }

            let mut output = File::create(o.file)?;
            let b = bf.lock().unwrap();
            b.write(&mut output)?;
        }
        Command::Check(o) => {
            let bloom_file = File::open(&o.file)?;
            let bf = Arc::new(std::sync::RwLock::new(BloomFilter::from_reader(
                bloom_file,
            )?));

            let mut handles = vec![];
            let files = o.inputs.clone();

            let batches = files.chunks(max(files.len() / o.jobs, 1));

            for batch in batches {
                let shared = Arc::clone(&bf);
                let batch: Vec<String> = batch.to_vec();

                let h = thread::spawn(move || {
                    let bf = shared
                        .read()
                        .map_err(|e| anyhow!("failed to lock mutex: {}", e))?;

                    for input in batch {
                        let in_file = std::fs::File::open(&input)?;

                        for line in std::io::BufReader::new(in_file).lines() {
                            let line = line?;
                            if o.verify {
                                if !bf.contains(&line) {
                                    println!("{line}: NOK")
                                }
                                continue;
                            }

                            if bf.contains(&line) {
                                println!("{line}: OK")
                            } else {
                                println!("{line}: NOK")
                            }
                        }
                    }

                    Ok::<(), anyhow::Error>(())
                });
                handles.push(h)
            }

            for h in handles {
                h.join().expect("failed to join thread")?;
            }
        }
        Command::Show(o) => {
            let bloom_file = File::open(&o.file)?;
            let b = BloomFilter::from_reader(bloom_file)?;
            println!("File: {}", o.file);
            println!("\tn (desired number of elements)={}", b.cap());
            println!("\tp (false positive rate)={}", b.proba());
            println!("\tk (number of hash fn)={}", b.n_hash_fn());
            println!("\tm (number of bits)={}", b.n_bits());
            println!("\tN (estimated number of elements)={}", b.count_estimate());
            println!("\tM (number of u64)={}", b.M);
            println!("\tLength of data={}", ByteSize::from_bytes(b.data.len()));
            println!(
                "\tSize of bloom filter={}",
                ByteSize::from_bytes(b.M as usize * size_of::<u64>())
            );
        }
    }
    Ok(())
}
