use std::{
    cmp::max,
    collections::HashSet,
    fs::File,
    io::{self, BufRead},
    sync::Arc,
    thread,
};

use anyhow::anyhow;
use clap::Parser;
use poppy::{
    bloom,
    utils::{is_power_of_2, time_it, time_it_once, ByteSize, Stats},
    BloomFilter, DEFAULT_VERSION,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

// When built with musl, the allocator slows down a lot insertion
// so we have to use jemalloc to get back to the expected performance
#[cfg(all(target_env = "musl", target_pointer_width = "64"))]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

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
    /// Benchmark the bloom filter. If the bloom filter
    /// behaves in an unexpected way, the benchmark fails.
    Bench(Benchmark),
    /// Show information about an existing bloom filter
    Show(Show),
}

#[derive(Debug, Parser)]
struct Create {
    /// Creates a specific version of Bloom Filter
    /// Set to 1 if cross-compatibility with other DCSO libraries
    #[clap(short,long, default_value_t=DEFAULT_VERSION)]
    version: u8,
    /// Capacity of the bloom filter
    #[clap(short, long, default_value = "10000")]
    capacity: u64,
    /// False positive rate
    #[clap(short, long, default_value = "0.01")]
    probability: f64,
    /// Optimize the bloom filter for a given number of element by respecting
    /// the target probability for the filter
    #[clap(short, long)]
    optimize: bool,
    /// List the possible values to optimize, sometimes a set of better parameters
    /// can be found in this list when wanting some specific optimization (size, speed)
    #[clap(long)]
    list_opts: Option<usize>,
    /// File to store the bloom filter
    file: Option<String>,
}

#[derive(Debug, Parser)]
struct Insert {
    /// Show progress information
    #[clap(short, long)]
    progress: bool,
    /// Reads input from stdin
    #[clap(long)]
    stdin: bool,
    /// Force data insertion, eventually breaking bloom filter FP rate
    #[clap(long)]
    force: bool,
    /// The number of jobs to use to insert into the bloom filter. The original
    /// filter is copied into the memory of each job so you can expect the memory
    /// of the whole process to be N times the size of the (uncompressed) bloom filter.
    #[clap(short, long, default_value = "2")]
    jobs: usize,
    /// File containing the bloom filter to update
    file: String,
    /// Input files containing one entry per line, if no files is specified
    /// entries to insert are read from stdin
    inputs: Vec<String>,
}

#[derive(Debug, Parser)]
struct Check {
    /// checks input from stdin
    #[clap(long)]
    stdin: bool,
    /// do not print anything to stdout
    #[clap(long)]
    silent: bool,
    /// Numbers of parallel jobs to check bloom filter
    #[clap(short, long, default_value = "2")]
    jobs: usize,
    /// Only show entries not in the bloom filter
    #[clap(long)]
    verify: bool,
    /// File containing the bloom filter
    file: String,
    /// Input files containing one entry per line, if no files is specified
    /// entries to check are read from stdin
    inputs: Vec<String>,
}

#[derive(Debug, Parser)]
struct Benchmark {
    /// Size (in MB) limiting the number of entries to use for
    /// the benchmark.
    #[clap(short, long, default_value_t = 100)]
    size_mb: usize,
    /// Number of runs to compute statistics
    #[clap(short, long, default_value_t = 5)]
    runs: u32,
    /// Allowed tolerance on false positive proba before raising error
    #[clap(short, long, default_value_t = 0.05)]
    fp_tol: f64,
    /// File containing the bloom filter.
    file: String,
    /// Input files containing one entry per line. To have a relevant
    /// benchmark, entries in benchmark must have been inserted in
    /// the bloom filter.
    inputs: Vec<String>,
}

#[derive(Debug, Parser)]
struct Show {
    /// File containing the bloom filter
    file: String,
}

fn show_bf_properties(b: &BloomFilter) {
    println!("\tversion                         : {}", b.version());
    println!("\tn (desired number of elements)  : {}", b.cap());
    println!("\tp (false positive rate)         : {}", b.proba());
    println!("\testimated p (computed for N)    : {:.6}", b.estimated_p());
    println!("\tk (number of hash fn)           : {}", b.n_hash_fn());
    println!("\tm (number of bits)              : {}", b.n_bits());
    println!("\tN (estimated number of elements): {}", b.count_estimate());
    println!("\tM (number of u64)               : {}", b.size_in_u64());
    println!(
        "\tOptimized                       : {}",
        is_power_of_2(b.n_bits())
    );
    println!(
        "\tLength of data                  : {}",
        ByteSize::from_bytes(b.data.len())
    );
    println!(
        "\tSize of bloom filter            : {}",
        ByteSize::from_bytes(b.size_in_bytes())
    );
}

fn main() -> Result<(), anyhow::Error> {
    let sub = Args::parse();

    match sub.command {
        Command::Create(o) => {
            if let Some(n) = o.list_opts {
                println!("The following values are proposed if you want to optimize");
                println!("the speed of your bloom filter. Note the difference between");
                println!("the wanted p and the estimated one. Estimated p is computed from");
                println!("the size of the filter and the number of element you intend to store");
                println!("in it.\n");

                println!("Unoptimized parameters:");
                // original size
                let os =
                    ByteSize::from_bits(BloomFilter::bit_size(o.capacity, o.probability) as usize);
                println!("\tcap={} proba={} size={}\n", o.capacity, o.probability, os);

                let opts = BloomFilter::possible_optimizations(o.capacity, o.probability);

                println!("Optimized parameters:");
                let mut i = 1;
                let mut m = HashSet::new();
                for opt in opts {
                    if i > n {
                        break;
                    }

                    let opt_s = ByteSize::from_bits(opt.bit_size() as usize);
                    // only show one line per size as other entries are irrelevant
                    if m.contains(&opt_s) {
                        continue;
                    }

                    let mut optimize_str = vec![];

                    optimize_str.push("speed");
                    if opt_s < os {
                        optimize_str.push("size")
                    }
                    if opt.estimate_p(o.capacity) < o.probability {
                        optimize_str.push("proba")
                    }

                    print!(
                        "\tcap={} proba={} est(proba)={:.6} size={} optimize={}",
                        opt.cap,
                        opt.proba,
                        opt.estimate_p(o.capacity),
                        opt_s,
                        optimize_str.join(","),
                    );
                    if i == 1 {
                        print!(" -> Default optimization")
                    }
                    println!();
                    m.insert(opt_s);
                    i += 1
                }
                return Ok(());
            }

            let mut f = File::create(o.file.expect("missing destination file"))?;

            let b = {
                if o.optimize {
                    let p = BloomFilter::optimize(o.capacity, o.probability)
                        .expect("cannot optimize bloom filter");

                    // control optimization parameter
                    if p.estimate_p(o.capacity) > o.probability {
                        Err(anyhow!(
                            "optimization does not respect the wanted probability"
                        ))?;
                    }

                    p.into_bloom_filter().with_version(o.version)?
                } else {
                    bloom!(o.capacity, o.probability).with_version(o.version)?
                }
            };

            b.write(&mut f)?;
        }
        Command::Insert(o) => {
            let bloom_file = File::open(&o.file)?;
            let bf = Arc::new(std::sync::Mutex::new(BloomFilter::from_reader(bloom_file)?));

            // if we pipe in entries via stdin
            if o.stdin || o.inputs.is_empty() {
                let mut bf = bf.lock().unwrap();
                for line in std::io::BufReader::new(io::stdin()).lines() {
                    if o.force {
                        bf.insert_bytes_unchecked(line?)
                    } else {
                        bf.insert_bytes(line?)?
                    }
                }
            }

            // processing files if any
            if !o.inputs.is_empty() {
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
                        for input in batch {
                            if o.progress {
                                println!("processing file: {input}");
                            }
                            let in_file = std::fs::File::open(&input)?;

                            for line in std::io::BufReader::new(in_file).lines() {
                                let line = line?;
                                if o.force {
                                    copy.insert_bytes_unchecked(&line)
                                } else {
                                    copy.insert_bytes(&line)?
                                }
                            }
                        }

                        let mut shared = shared
                            .lock()
                            .map_err(|e| anyhow!("failed to lock mutex: {}", e))?;
                        shared.union(&copy)?;

                        Ok::<(), anyhow::Error>(())
                    });
                    handles.push(h)
                }

                for h in handles {
                    h.join().expect("failed to join thread")?;
                }
            }

            let mut output = File::create(o.file)?;
            bf.lock().unwrap().write(&mut output)?;
        }

        Command::Check(o) => {
            let bloom_file = File::open(&o.file)?;
            let bf = Arc::new(std::sync::RwLock::new(BloomFilter::from_reader(
                bloom_file,
            )?));

            if o.stdin || o.inputs.is_empty() {
                let t = time_it_once(|| {
                    let bf = bf.read().unwrap();
                    for line in io::stdin().lines() {
                        let line = line.unwrap();
                        let ok = bf.contains_bytes(&line);

                        if o.verify {
                            if !ok {
                                println!("{line}: NOK")
                            }
                            continue;
                        }

                        if ok && !o.silent {
                            println!("{line}")
                        }
                    }
                });

                eprintln!("Check time: {t:?}");
            }

            if !o.inputs.is_empty() {
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
                                let ok = bf.contains_bytes(&line);

                                if o.verify {
                                    if !ok {
                                        println!("{line}: NOK")
                                    }
                                    continue;
                                }

                                if ok && !o.silent {
                                    println!("{line}")
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
        }
        Command::Bench(o) => {
            let mut rng: StdRng = SeedableRng::from_seed([42; 32]);
            let bloom_file = File::open(&o.file)?;
            let b = BloomFilter::from_reader(bloom_file)?;
            let mut entries = HashSet::new();
            let mut dataset_size = 0usize;
            let limit = ByteSize::from_mb(o.size_mb);

            for line in io::stdin().lines() {
                let line = line?;
                if ByteSize::from_bytes(dataset_size) > limit {
                    break;
                }
                if !b.contains_bytes(&line) {
                    return Err(anyhow!(
                        "bloom filter to benchmark must contain all entries"
                    ));
                };
                if !entries.contains(&line) {
                    dataset_size += line.as_bytes().len();
                    entries.insert(line);
                }
            }

            println!("File: {}", o.file);
            show_bf_properties(&b);

            println!(
                "\nQuery performance: dataset-size={}",
                ByteSize::from_bytes(dataset_size)
            );

            for mut_prob in (10..=100).step_by(10) {
                let mut tmp = entries.iter().cloned().collect::<Vec<String>>();
                let mutated_lines = tmp
                    .iter_mut()
                    .map(|e| {
                        let mut mutated = false;
                        if rng.gen_range(0..=100) < mut_prob {
                            mutated = true;
                            unsafe { e.as_bytes_mut() }
                                .iter_mut()
                                .for_each(|b| *b ^= rng.gen_range(0..=255));
                        }
                        (mutated, e.clone())
                    })
                    .collect::<Vec<(bool, String)>>();

                let mut stats = Stats::new();
                let query_dur = time_it(
                    || {
                        mutated_lines.iter().for_each(|(m, l)| {
                            let is_in_bf = b.contains_bytes(l);
                            // data has been mutated
                            if *m {
                                if is_in_bf {
                                    stats.inc_fp()
                                } else {
                                    stats.inc_tn()
                                }
                            }
                        })
                    },
                    o.runs,
                );

                println!(
                    "\tcondition: {}% of queried values are in filter",
                    100 - mut_prob
                );
                println!("\tquery duration: {:?}", query_dur);
                println!(
                    "\tquery speed: {:.1} entries/s -> {:.1} MB/s",
                    stats.total() as f64 / query_dur.as_secs_f64(),
                    ByteSize::from_bytes(dataset_size).in_mb() / query_dur.as_secs_f64()
                );
                println!("\tfp rate = {:3}", stats.fp_rate());
                println!();

                if stats.fp_rate() > b.proba() + b.proba() * o.fp_tol {
                    return Err(anyhow!("bloom filter fp rate is not respected"));
                }
            }
        }
        Command::Show(o) => {
            let bloom_file = File::open(&o.file)?;
            let b = BloomFilter::from_reader(bloom_file)?;
            println!("File: {}", o.file);
            show_bf_properties(&b);
        }
    }
    Ok(())
}
