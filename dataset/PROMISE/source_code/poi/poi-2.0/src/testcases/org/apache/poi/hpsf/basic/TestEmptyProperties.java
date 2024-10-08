package org.apache.poi.hpsf.basic;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import junit.framework.Assert;
import junit.framework.TestCase;

import org.apache.poi.hpsf.HPSFException;
import org.apache.poi.hpsf.MarkUnsupportedException;
import org.apache.poi.hpsf.NoPropertySetStreamException;
import org.apache.poi.hpsf.PropertySet;
import org.apache.poi.hpsf.PropertySetFactory;
import org.apache.poi.hpsf.SummaryInformation;
import org.apache.poi.hpsf.UnexpectedPropertySetTypeException;

/**
 * <p>Test case for OLE2 files with empty properties. An empty property's type
 * is {@link Variant.VT_EMPTY}.</p>
 *
 * @author Rainer Klute <a
 * href="mailto:klute@rainer-klute.de">&lt;klute@rainer-klute.de&gt;</a>
 * @since 2003-07-25
 * @version $Id$
 */
public class TestEmptyProperties extends TestCase
{

    /**
     * <p>This test file's summary information stream contains some empty
     * properties.</p>
     */
    static final String POI_FS = "TestCorel.shw";

    static final String[] POI_FILES = new String[]
        {
            "PerfectOffice_MAIN",
            "\005SummaryInformation",
            "Main"
        };

    POIFile[] poiFiles;



    /**
     * <p>Constructor</p>
     * 
     * @param name The name of the test case
     */
    public TestEmptyProperties(final String name)
    {
        super(name);
    }



    /**
     * <p>Read a the test file from the "data" directory.</p>
     *
     * @exception FileNotFoundException if the file containing the test data
     * does not exist
     * @exception IOException if an I/O exception occurs
     */
    public void setUp() throws FileNotFoundException, IOException
    {
        final File dataDir =
            new File(System.getProperty("HPSF.testdata.path"));
        final File data = new File(dataDir, POI_FS);

        poiFiles = Util.readPOIFiles(data);
    }



    /**
     * <p>Checks the names of the files in the POI filesystem. They
     * are expected to be in a certain order.</p>
     * 
     * @exception IOException if an I/O exception occurs
     */
    public void testReadFiles() throws IOException
    {
        String[] expected = POI_FILES;
        for (int i = 0; i < expected.length; i++)
            Assert.assertEquals(poiFiles[i].getName(), expected[i]);
    }



    /**
     * <p>Tests whether property sets can be created from the POI
     * files in the POI file system. This test case expects the first
     * file to be a {@link SummaryInformation}, the second file to be
     * a {@link DocumentSummaryInformation} and the rest to be no
     * property sets. In the latter cases a {@link
     * NoPropertySetStreamException} will be thrown when trying to
     * create a {@link PropertySet}.</p>
     * 
     * @exception IOException if an I/O exception occurs
     */
    public void testCreatePropertySets() throws IOException
    { 
        Class[] expected = new Class[]
            {
                NoPropertySetStreamException.class,
                SummaryInformation.class,
                NoPropertySetStreamException.class
            };
        for (int i = 0; i < expected.length; i++)
        {
            InputStream in = new ByteArrayInputStream(poiFiles[i].getBytes());
            Object o;
            try
            {
                o = PropertySetFactory.create(in);
            }
            catch (NoPropertySetStreamException ex)
            {
                o = ex;
            }
            catch (UnexpectedPropertySetTypeException ex)
            {
                o = ex;
            }
            catch (MarkUnsupportedException ex)
            {
                o = ex;
            }
            in.close();
            Assert.assertEquals(o.getClass(), expected[i]);
        }
    }



    /**
     * <p>Tests the {@link PropertySet} methods. The test file has two
     * property sets: the first one is a {@link SummaryInformation},
     * the second one is a {@link DocumentSummaryInformation}.</p>
     * 
     * @exception IOException if an I/O exception occurs
     * @exception HPSFException if an HPSF operation fails
     */
    public void testPropertySetMethods() throws IOException, HPSFException
    {
        byte[] b = poiFiles[1].getBytes();
        PropertySet ps =
            PropertySetFactory.create(new ByteArrayInputStream(b));
        SummaryInformation s = (SummaryInformation) ps;
        assertNull(s.getTitle());
        assertNull(s.getSubject());
        assertNotNull(s.getAuthor());
        assertNull(s.getKeywords());
        assertNull(s.getComments());
        assertNotNull(s.getTemplate());
        assertNotNull(s.getLastAuthor());
        assertNotNull(s.getRevNumber());
        assertNull(s.getEditTime());
        assertNull(s.getLastPrinted());
        assertNull(s.getCreateDateTime());
        assertNull(s.getLastSaveDateTime());
        assertEquals(s.getPageCount(), 0);
        assertEquals(s.getWordCount(), 0);
        assertEquals(s.getCharCount(), 0);
        assertNull(s.getThumbnail());
        assertNull(s.getApplicationName());
    }



    /**
     * <p>Runs the test cases stand-alone.</p>
     * 
     * @param args the command-line arguments (unused)
     * 
     * @exception Throwable if any exception or error occurs
     */
    public static void main(final String[] args) throws Throwable
    {
        System.setProperty("HPSF.testdata.path",
                           "./src/testcases/org/apache/poi/hpsf/data");
        junit.textui.TestRunner.run(TestBasic.class);
    }

}
