/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @microservice:  export-client
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/
package org.edgexfoundry.controller.integration.web;

import static org.edgexfoundry.test.data.RegistrationData.TEST_ADDR_NAME;
import static org.edgexfoundry.test.data.RegistrationData.TEST_NAME;
import static org.edgexfoundry.test.data.RegistrationData.checkTestData;
import static org.edgexfoundry.test.data.RegistrationData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.controller.ExportRegistrationController;
import org.edgexfoundry.dao.ExportRegistrationRepository;
import org.edgexfoundry.domain.export.ExportCompression;
import org.edgexfoundry.domain.export.ExportDestination;
import org.edgexfoundry.domain.export.ExportEncryption;
import org.edgexfoundry.domain.export.ExportFormat;
import org.edgexfoundry.domain.export.ExportRegistration;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({ RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class })
public class ExportRegistrationControllerTest {

	@Autowired
	ExportRegistrationController controller;

	@Autowired
	ExportRegistrationRepository repos;

	private String id;

	@Before
	public void setup() {
		ExportRegistration reg = newTestInstance();
		repos.save(reg);
		id = reg.getId();
		assertNotNull("Test export registration not created properly", id);
	}

	@After
	public void cleanup() throws Exception {
		resetControllerRepos();
		repos.deleteAll();
	}

	@Test
	public void testExportRegistration() {
		checkTestData(controller.exportRegistration(id), id);
	}

	@Test
	public void testEncryptionOptions() {
		String[] opts = controller.referenceOptions("algorithms");
		List<String> options = Arrays.asList(opts);
		for (ExportEncryption option : ExportEncryption.values()) {
			assertTrue("Export encryption option not included in list", options.contains(option.name()));
		}
	}

	@Test
	public void testCompressions() {
		String[] opts = controller.referenceOptions("compressions");
		List<String> options = Arrays.asList(opts);
		for (ExportCompression option : ExportCompression.values()) {
			assertTrue("Export compression option not included in list", options.contains(option.name()));
		}
	}

	@Test
	public void testDestinations() {
		String[] opts = controller.referenceOptions("destinations");
		List<String> options = Arrays.asList(opts);
		for (ExportDestination option : ExportDestination.values()) {
			assertTrue("Export destination option not included in list", options.contains(option.name()));
		}
	}

	@Test
	public void testFormats() {
		String[] opts = controller.referenceOptions("formats");
		List<String> options = Arrays.asList(opts);
		for (ExportFormat option : ExportFormat.values()) {
			assertTrue("Export format option not included in list", options.contains(option.name()));
		}
	}

	@Test
	public void testExportRegistrationWithBadId() {
		assertNull("Fetch of export registration with bad id returned a registration",
				controller.exportRegistration("unknownid"));
	}

	@Test(expected = ServiceException.class)
	public void testExportRegistrationServiceException() throws Exception {
		unsetControllerRepos();
		controller.exportRegistration(id);
	}

	@Test
	public void testExportRegistrations() {
		List<ExportRegistration> regs = controller.exportRegistrations();
		assertFalse("Fetch of export registrations is empty", regs.isEmpty());
		checkTestData(regs.get(0), id);
	}

	@Test(expected = ServiceException.class)
	public void testExportRegistrationsServiceException() throws Exception {
		unsetControllerRepos();
		controller.exportRegistrations();
	}

	@Test
	public void testExportRegistrationByName() {
		checkTestData(controller.exportRegistrationByName(TEST_NAME), id);
	}

	@Test
	public void testExportRegistrationWithBadName() {
		assertNull("Fetch of export registration with bad name returned a registration",
				controller.exportRegistrationByName("unknownname"));
	}

	@Test(expected = ServiceException.class)
	public void testExportRegistrationByNameServiceException() throws Exception {
		unsetControllerRepos();
		controller.exportRegistrationByName(TEST_NAME);
	}

	@Test
	public void testAdd() {
		ExportRegistration reg = newTestInstance();
		// names must be unique
		reg.setName("newname");
		reg.getAddressable().setName("newaddressname");
		String regId = controller.add(reg);
		assertNotNull("New registration not added correctly", regId);
		// put the names back so it can be checked
		reg.setName(TEST_NAME);
		reg.getAddressable().setName(TEST_ADDR_NAME);
		checkTestData(reg, regId);
	}

	@Test(expected = ServiceException.class)
	public void testAddWithSameName() {
		ExportRegistration reg = newTestInstance();
		controller.add(reg);
	}

	@Test(expected = ServiceException.class)
	public void testAddServiceException() throws Exception {
		unsetControllerRepos();
		ExportRegistration reg = newTestInstance();
		// name must be unique
		reg.setName("newname");
		controller.add(reg);
	}

	@Test
	public void testUpdate() {
		ExportRegistration reg = new ExportRegistration();
		reg.setId(id);
		reg.setOrigin(123456);
		controller.update(reg);
		ExportRegistration reg2 = repos.findOne(id);
		assertEquals("Update did not work properly", 123456, reg2.getOrigin());
	}

	@Test(expected = NotFoundException.class)
	public void testUpdateWithBadId() {
		ExportRegistration reg = new ExportRegistration();
		reg.setId("badid");
		reg.setOrigin(123456);
		controller.update(reg);
	}

	@Test(expected = ServiceException.class)
	public void testUpdateServiceException() throws Exception {
		unsetControllerRepos();
		ExportRegistration reg = new ExportRegistration();
		reg.setId(id);
		reg.setOrigin(123456);
		controller.update(reg);
	}

	@Test
	public void testDelete() {
		assertTrue("Delete did not work properly", controller.delete(id));
		assertNull("After delete, registration still available", repos.findOne(id));
	}

	@Test(expected = ServiceException.class)
	public void testDeleteServiceException() throws Exception {
		unsetControllerRepos();
		controller.delete(id);
	}

	@Test(expected = NotFoundException.class)
	public void testDeleteWithBadId() throws Exception {
		controller.delete("badid");
	}

	@Test
	public void testDeleteByName() {
		assertTrue("Delete by name did not work properly", controller.deleteByName(TEST_NAME));
		assertNull("After delete by name, registration still available", repos.findOne(id));
	}

	@Test(expected = ServiceException.class)
	public void testDeletebyNameServiceException() throws Exception {
		unsetControllerRepos();
		controller.deleteByName(TEST_NAME);
	}

	@Test(expected = NotFoundException.class)
	public void testDeleteWithBadName() throws Exception {
		controller.deleteByName("badname");
	}

	// use Java reflection to unset controller's repos
	private void unsetControllerRepos() throws Exception {
		Class<?> controllerClass = controller.getClass();
		Field temp = controllerClass.getDeclaredField("repos");
		temp.setAccessible(true);
		temp.set(controller, null);
	}

	// use Java reflection to reset controller's repos
	private void resetControllerRepos() throws Exception {
		Class<?> controllerClass = controller.getClass();
		Field temp = controllerClass.getDeclaredField("repos");
		temp.setAccessible(true);
		temp.set(controller, repos);
	}

}
